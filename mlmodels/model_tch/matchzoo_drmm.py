import matchzoo as mz

class DRMM(mz.models.DRMM):
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
        self.params['mask_value'] = model_pars['mask_value']
        self.params['embedding'] = self.embedding_matrix
        self.params['hist_bin_size'] = 30
        self.params['mlp_num_layers'] = 1
        self.params['mlp_num_units'] = 10
        self.params['mlp_num_fan_out'] = 1
        self.params['mlp_activation_func'] = 'tanh'
        self.build()
        
    def prepare_data(self, data_pars, train_pack_raw, dev_pack_raw, test_pack_raw):
        preprocessor = self.get_default_preprocessor()
        train_pack_processed = preprocessor.fit_transform(train_pack_raw)
        dev_pack_processed = preprocessor.transform(dev_pack_raw)
        test_pack_processed = preprocessor.transform(test_pack_raw)

        glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        embedding_matrix = glove_embedding.build_matrix(term_index)
        l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
        self.embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

        histgram_callback = mz.dataloader.callbacks.Histogram(
            self.embedding_matrix, bin_size=30, hist_mode='LCH'
        )

        train_mode = data_pars["mode"] if "mode" in data_pars else None
        train_num_dup = data_pars["num_dup"] if "num_dup" in data_pars else None
        train_num_neg = data_pars["num_neg"] if "num_neg" in data_pars else None
        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode=train_mode,
            num_dup=train_num_dup,
            num_neg=train_num_neg,
            callbacks=[histgram_callback]
        )
        testset = mz.dataloader.Dataset(
            data_pack=test_pack_processed,
            callbacks=[histgram_callback]
        )

        padding_callback = mz.models.DRMM.get_default_padding_callback()

        train_resample = data_pars["train_resample"] if "train_resample" in data_pars else False
        trainloader = mz.dataloader.DataLoader(
            device='cpu',
            dataset=trainset,
            batch_size=data_pars["train_batch_size"],
            stage='train',
            resample=train_resample,
            callback=padding_callback
        )
        testloader = mz.dataloader.DataLoader(
            dataset=testset,
            batch_size=data_pars["test_batch_size"],
            stage='dev',
            callback=padding_callback
        )
        return trainloader, testloader
