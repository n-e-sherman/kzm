import pandas as pd
import pickle
import os

class Repository:
    
    def __init__(self, options):
        self.options = options
        self._set_io_paths()
        
    def _set_io_paths(self):
        self.data_dir = self.options.get('data_dir','.data/')
        self.results_dir = self.options.get('res_dir','.results/')
        self.log_dir = self.options.get('log_dir','.log/')
        
class Repository_Pickle(Repository):
    
    def __init__(self, options):
        super().__init__(options)
        
    def write(self, file, df):
        df.to_csv(self.results_dir + file, index=False)
    
    def save(self, file, data):
        with open(self.data_dir + file, 'wb') as f:
            for k,v in data.items():
                pickle.dump(v, f)
                
    def log(self, file, df):
        df.to_csv(self.log_dir + file, index=False)
        
    def load(self, file, data):
        if not os.path.exists(self.data_dir + file):
            return False
        try:
            with open(self.data_dir + file, 'rb') as f:
                for key in data:
                    data[key] = pickle.load(f)
            return True
        except:
            return False