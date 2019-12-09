#!/usr/bin/env python
# -*- coding:utf-8 _*-
# Time: 2019/12/06
# Author: zcy

import requests
from flask import Flask, request, jsonify
from detect import *

app = Flask(__name__)
formats = ['JPEG', 'PNG', 'BMP']


@app.route('/detect', methods=['POST'])
def start_service():
    detector = Detector()

    jo = request.json
    img_url = jo.get('img_url')
    print('detecting image: ' + img_url)
    data = {
        'result': 'fail'
    }

    with open('tmp_img', 'wb') as f:
        res = requests.get(img_url)
        if res.status_code != 404:
            f.write(res.content)
            try:
                img_format = Image.open('tmp_img').format
                if img_format not in formats:
                    raise Exception
            except:
                return jsonify(data)
        else:
            return data

    with torch.no_grad():
        results = detector.detect('tmp_img')
    if results is None or len(results) == 0:
        return jsonify(data)

    data['result'] = 'success'
    data['box'] = []
    for result in results:
        data['box'].append({
            'confidence': float(result[-1]),
            'coordinates': result[:4]
        })
        print(result)

    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=13333)
