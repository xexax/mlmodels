import matchzoo as mz

class Bert(mz.models.Bert):
    def __init__(self,
        model_pars, 
        data_pars,
        task=None,
        train_pack_raw=None,
        dev_pack_raw=None,
        test_pack_raw=None 
    ):
        self.trainloader, self.validloader = self.prepare_data(
            data_pars, train_pack_raw, dev_pack_raw, test_pack_raw)
        super().__init__()
        self.params['task'] = task
        self.params['mode'] = model_pars['mode']
        self.params['dropout_rate'] = model_pars['dropout_rate']
        self.build()
    
    def prepare_data(self, data_pars, train_pack_raw, dev_pack_raw, test_pack_raw):
        preprocessor = self.get_default_preprocessor()
        train_pack_processed = preprocessor.transform(train_pack_raw)
        dev_pack_processed = preprocessor.transform(dev_pack_raw)
        test_pack_processed = preprocessor.transform(test_pack_raw)

        train_mode = data_pars["mode"] if "mode" in data_pars else None
        train_sort = data_pars["train_sort"] if "train_sort" in data_pars else False
        
        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode=train_mode,
            batch_size=data_pars["train_batch_size"],
            sort=train_sort,
        )
        testset = mz.dataloader.Dataset(
            data_pack=test_pack_processed,
            batch_size=data_pars["test_batch_size"],
        )
        padding_callback = self.get_default_padding_callback()
        trainloader = mz.dataloader.DataLoader(
            dataset=trainset,
            stage='train',
            callback=padding_callback
        )
        testloader = mz.dataloader.DataLoader(
            dataset=testset,
            stage='dev',
            callback=padding_callback
        )
        return trainloader, testloader
