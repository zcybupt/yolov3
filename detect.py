from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import cv2


class Detector:
    img_size: int

    def __init__(self):
        cfg = 'cfg/yolov3-1cls.cfg'
        weights = 'weights/yolov3-voc.weights'
        self.data = 'cfg/watermark.data'
        self.img_size = 416
        self.conf_thres = 0.3
        self.nms_thres = 0.5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.half = True if torch.cuda.is_available() else False

        # Initialize
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.device)

        # Initialize model
        self.model = Darknet(cfg, self.img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(weights, map_location=self.device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self.model, weights)

    def detect(self, filepath):
        # Eval mode
        self.model.to(self.device).eval()

        # Export mode
        if ONNX_EXPORT:
            img = torch.zeros((1, 3) + self.img_size)  # (1, 3, 320, 192)
            torch.onnx.export(self.model, img, 'weights/export.onnx', verbose=False, opset_version=11)

            # Validate exported models
            import onnx
            model = onnx.load('weights/export.onnx')  # Load the ONNX model
            onnx.checker.check_model(model)  # Check that the IR is well formed
            print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
            return

        # Half precision
        half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            self.model.half()

        # Get classes and colors
        classes = load_classes(parse_data_cfg(self.data)['names'])

        # Run inference
        t0 = time.time()
        img0 = cv2.imread(filepath)
        if img0 is None:
            return None

        # Padded resize
        img = letterbox(img0, new_shape=self.img_size)[0]

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(self.device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self.model(img)[0]

        if self.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.nms_thres)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', img0

            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, classes[int(c)])  # add to string

            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Save results (image with detections)
            if pred[0] is not None:
                watermaks = np.asarray(pred[0])
                results = []
                for watermak in watermaks:
                    result = [int(i) for i in watermak[:4]]
                    result.append(watermak[4])
                    results.append(result)
                return results
