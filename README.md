# DataMining

As a part of Data Mining course in masters I have here create a recommendation system using hybrid method.The hybrid here uses model based  collaborative filtering as the base. 
To improve the performance of model based CF I performed data  preprocessing so as to have more information from the same data. I have performed Principal Component  Analysis to increase data interpretability and dimension reduction. 
I used only the 10 most important  features thus generated. Over model based CF, I then utitlised item based collaborative filtering. I  used prediction generared by item based as one of the feature to model based CF.

Error Distribution:
>=0 and <1: 105773
>=1 and <2: 34123
>=2 and <3: 6310
>=3 and <4: 790
>=4: 0

RSME: 0.9782285911303371
Duration: 861.13
