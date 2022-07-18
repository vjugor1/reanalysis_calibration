# `.ipynb`s overview
## `era5.ipynb`
- Connects to Google Earth Engine (GEE), retrieves climate reanalysis data, matches reanalysis data with weather stations
- Saves matched data into corresponding folders (by station names)
- Change `vicinity_degree` to increase `X` shape that will be fed into CNN
## `train_model.ipynb`
- Example of CNN that could be used on such a data
- Change model that is written in `src/models/WindCNN.py` 
- Monitor performance using TensorBoard

# Where to get weather stations data?
- Demo variant can be donwloaded from here: https://drive.google.com/drive/folders/1yjibnllNTtVmP6w9fO3GByQM0wIiwiVv?usp=sharing
- Put that into `data` folder
- Download more stations from here (Раздел БД: Сроки, источник данных: SROK8C): http://meteo.ru/data/163-basic-parameters   
- Do not download 'Архив с полными данными раздела: Srok8c.zip Размер архива: 3.82 GB'! There data inside is corrupted!
# How to work with GEE?
- Go to https://earthengine.google.com/
- Register using google email account, declare 'eduational purposes'
- Wait...
- Go to `era5.ipynb`
- Run cell with `ee.Authenticate()`, follow instructions
- Authenticate, as google will redirect you
- Have fun!