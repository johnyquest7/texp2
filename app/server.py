import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from dummy_function import *
from PIL import Image


path = Path(__file__).parent


app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
#app.mount('/static', StaticFiles(directory=static_root_absolute))
app.mount('/static', StaticFiles(directory='app/static'))
app.mount('/tmp', StaticFiles(directory='/tmp'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)




@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = predict(img)
    #similar1= open_image(path / 'static' / 'similar1.JPG')
    similar1 = Image.open(path / 'static' / 'similar1.JPG')	
    similar2 = Image.open(path / 'static' / 'similar2.JPG')
    similar3 = Image.open(path / 'static' / 'similar3.JPG')
    similar1.save('/tmp/similar1.JPG')
    similar2.save('/tmp/similar2.JPG')
    similar3.save('/tmp/similar3.JPG')
    return JSONResponse({'result': prediction})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=8080, log_level="info")
