FROM waggle/plugin-base:1.1.1-ml

COPY requirements.txt /app/
COPY unet /app/unet
COPY app.py unet_module.py color_map.py /app/

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r /app/requirements.txt

ADD https://web.lcrc.anl.gov/public/waggle/models/waterdepth/model.pth /app/model.pth

WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/app.py"]
