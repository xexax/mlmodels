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
    
    def get_optimized_parameters(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 5e-5},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        return optimizer_grouped_parameters

    def prepare_data(self, data_pars, train_pack_raw, dev_pack_raw, test_pack_raw):
        preprocessor = self.get_default_preprocessor()
        train_pack_processed = preprocessor.transform(train_pack_raw)
        dev_pack_processed   = preprocessor.transform(dev_pack_raw)
        test_pack_processed  = preprocessor.transform(test_pack_raw)

        train_mode = data_pars["mode"] if "mode" in data_pars else None
        train_num_dup = data_pars["num_dup"] if "num_dup" in data_pars else None
        train_num_neg = data_pars["num_neg"] if "num_neg" in data_pars else None
        train_resample = data_pars["train_resample"] if "train_resample" in data_pars else False
        train_sort = data_pars["train_sort"] if "train_sort" in data_pars else False
        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode=train_mode,
            num_dup=train_num_dup,
            num_neg=train_num_neg,
            batch_size=data_pars["train_batch_size"],
            resample=True,
            sort=False,
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
