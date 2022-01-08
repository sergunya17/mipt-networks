# mipt-networks

## Запуск

Адрес сервера: `172.200.17.30`

```bash
docker-compose up
```

Пример запроса:
```bash
curl -F "imageFile=@img.jpg" "http://172.200.17.30:5000/image/recognize/detect-objects?foreign_lang=es&native_lang=ru"
```