YuGiOCR
==========

OCR based YuGiOh card detector.

Examples in different lightning scenarios and positioning scenarios:

***Cold light***

![Three cards - cold light](https://i.imgur.com/XeG80jr.png)

***Warm light***

![Two cards - warm light](https://i.imgur.com/07CAMpR.png)

Detection fails mostly when the full card shape is somewhat occluded as you can see in the below example.

![Failure example](https://i.imgur.com/1eOyxxD.png)

![Failure example 2](https://i.imgur.com/xoS37fa.png)


Dependencies
------------
* [`tesseract`](https://github.com/tesseract-ocr/)
* `python` and `pip` then run `pip install -r requirements.txt`

Card data
------------
`cardinfo.php` has been scraped via instructions [`here`](https://ygoprodeck.com/api-guide/).

It contains up to date information on all the released cards, so it should be occasionally updated if the recognition on new cards fails a lot.

Usage
------------
```
usage: detector_final.py [-h] [--image IMAGE] [--tesseract TESSERACT] [--visualize]

options:
  -h, --help            show this help message and exit
  --image IMAGE         path to image
  --tesseract TESSERACT
                        path to tesseract.exe
  --visualize           show intermediate steps
```

`--visualize` is very optional, useful for bugtesting purposes.

Future Work:
------------
* Improve 4-point contour detection mechanism
* Optimize the code speed for real-time video usage (remove unnecessary contour detection, use a faster Tesseract wrapper)


Related GitHub Projects:
------------
[`Yugioh card scanner`](https://github.com/theDataFox/yugioh-card-scanner)
