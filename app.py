#!/usr/bin/env python
# -*- coding:utf-8 _*-
# Time: 2019/12/06
# Author: zcy

import requests
from flask import Flask, request, jsonify
from detect import detect

app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def detect_img():
    jo = request.json
    img_url = jo.get('img_url')
    data = {
        'result': 'fail'
    }

    with open('tmp.jpg', 'wb') as f:
        res = requests.get(img_url)
        if res.status_code == 404:
            f.write(res.content)
        else:
            return data

    results = detect(net, meta, 'tmp.jpg'.encode('utf-8'))
    if len(results) == 0:
        return jsonify(data)

    data['result'] = 'success'
    data['box'] = []
    for result in results:
        data['box'].append({
            'confidence': result[1],
            'coordinates': [round(result[2][0] - result[2][2] / 2),
                            round(result[2][1] - result[2][3] / 2),
                            round(result[2][0] + result[2][2] / 2),
                            round(result[2][1] + result[2][3] / 2)]
        })
        print(result)

    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=23333)
