

#pip install numpy==1.18.2  pillow==6.2.1   && wget https://raw.githubusercontent.com/arita37/mlmodels/dev/requirements_fake.txt  && pip install -r requirements_fake.txt   && pip install  https://github.com/arita37/mlmodels/archive/dev.zip 




### Deps
pip install numpy==1.18.2  pillow==6.2.1  

mkdir z  ;  cd z && git clone https://github.com/arita37/mlmodels.git  ; cd mlmodels && pip install -r requirements.txt  ; pip install -r requirements_fake.txt  


### Extra for Colab
pip install torchvision==0.4.0



cd z/mlmodels && pip install -e .  --no-deps

