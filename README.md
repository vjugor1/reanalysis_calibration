# `.ipynb`s overview
## `era5.ipynb`
- Connects to Google Earth Engine (GEE), retrieves climate reanalysis data, matches reanalysis data with weather stations
- Saves matched data into corresponding folders (by station names)
- Change `vicinity_degree` to increase `X` shape that will be fed into CNN
## `train_model.ipynb`
- Example of CNN that could be used on such a data
- Change model that is written in `src/models/WindCNN.py` 
- Monitor performance using TensorBoard

# How to work with GEE?
- Go to https://earthengine.google.com/
- Register using google email account, declare 'eduational purposes'
- Wait...
- Go to `era5.ipynb`
- Run cell with `ee.Authenticate()`, follow instructions
- Authenticate, as google will redirect you
- Have fun!