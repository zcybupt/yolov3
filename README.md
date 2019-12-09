## 基于 Yolo v3 的 flask 在线检测接口

### 输入

POST

```html
http://[hostname]:13333/detect
```

Body

```javascript
{
	"img_url": "url_to_image"
}
```

### 输出

成功

```javascript
{
    "box": [
        {
            "confidence": 0.8726925849914551, # 置信度
            "coordinates": [
                95,  # xmin
                635, # ymin
                390, # xmax
                941  # ymax
            ]
        }
    ],
    "result": "success"
}
```

失败

```javascript
{
    'result': 'fail'
}
```