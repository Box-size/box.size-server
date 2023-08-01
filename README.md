# Box.size-server

사진에서 박스를 탐색하고, 실제 박스의 가로, 세로, 높이를 계산합니다.

Box.size의 FastAPI 백엔드 서버입니다.

> CJ대한통운 미래기술 챌린지 2023(Box.size 팀)

## Environment
> using python 3.11.2

### Installation
* Linux  
    ```sh
    $ source ./activate_venv.sh
    (.venv) $ pip install -r requirements.txt
    (.venv) $ python main.py
    ```

The server operates on [http://localhost:8000](http://localhost:8000).

you can watch [API DOCS in SWAGGER UI](http://52.79.88.247:8000/docs)


## 박스 길이 측정

### Run

```sh
$ python modules/box.py
```

### Detail

box.py내에서 Image.open()내의 인자를 변화시켜 다른 박스사진으로 테스트 해 볼 수 있습니다.