# Related repositories

To see how the dataset was build [head to this repository](https://github.com/Alex-Gombos/SaxParser)

Too see how the model was train [head to this repository](https://github.com/Alex-Gombos/Fine-tuning)

# How to run

After unzipping the project, run the following commands in the terminal to install all neccesarry dependecies:

## If you dont already have Python installed, simply download and install it from this link:

https://www.python.org/downloads/windows/

## Install PIP

curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

python get-pip.py

## Install all necessary packages with pip

After navigating to the location of the unzipped folder run these command:

pip install -r requirements.txt

This will install all dependencies listed in the txt file

## Running the application

In the same location, once all dependencies are installed, to start the application enter the command:

py app.py OR python app.py

depending or the installation of python on your computer

After that, in the terminal there will be outputed a http address where the application will run locally. This is the address in my case:

http://127.0.0.1:5000

From the web interface you can either enter text, or upload PDFs which are to be labeld. There is a drug leaflet included in this zipped folder for you to try out, called PRO_8889_26.04.16
