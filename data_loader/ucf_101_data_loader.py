from base.base_data_loader import BaseDataLoader



"""Moram priznati da nemam osjećaja u ovom trenutku o veličini vašeg dataseta i očekivanom trajanju učenja, 
ali da, vjerujem da se, u slučaju potrebe, može ograničiti broj frameova (možda čak uzimati svaki n-ti frame, 
tako da ostane sačuvana informacija o cijeloj aktivnosti u danom videu, samo u malo grubljoj "vremenskoj rezoluciji"). 
Predlažem da naprosto probate neki takav pristup pa ćemo vidjeti kamo će nas to odvesti."""
class Ucf101DataLoader(BaseDataLoader):

    def __init__(self, config: dict, ):
        super().__init__(config)

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass