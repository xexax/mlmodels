import matchzoo as mz

class DRMM(mz.models.DRMM):
    def __init__(self, model_pars, task=None):
        super().__init__()
        self.params['task'] = task
        self.params['mask_value'] = model_pars["mask_value"]
        self.params['embedding'] = get_embedding_matrix()
        self.params['hist_bin_size'] = model_pars["hist_bin_size"]
        self.params['mlp_num_layers'] = model_pars["mlp_num_layers"]
        self.params['mlp_num_units'] = model_pars["mlp_num_units"]
        self.params['mlp_num_fan_out'] = model_pars["mlp_num_fan_out"]
        self.params['mlp_activation_func'] = model_pars["mlp_activation_func"]
        self.model.build()

    def get_embedding_matrix(self):
        preprocessor = self.get_default_preprocessor()
        glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        embedding_matrix = glove_embedding.build_matrix(term_index)
        l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
        embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

    def preprocess_dataset(self):
        train_pack_processed = preprocessor.fit_transform(train_pack_raw)
        dev_pack_processed = preprocessor.transform(dev_pack_raw)
        test_pack_processed = preprocessor.transform(test_pack_raw)

        glove_embedding = mz.datasets.embeddings.load_glove_embedding(dimension=300)
        term_index = preprocessor.context['vocab_unit'].state['term_index']
        embedding_matrix = glove_embedding.build_matrix(term_index)
        l2_norm = np.sqrt((embedding_matrix * embedding_matrix).sum(axis=1))
        embedding_matrix = embedding_matrix / l2_norm[:, np.newaxis]

        histgram_callback = mz.dataloader.callbacks.Histogram(
            embedding_matrix, bin_size=30, hist_mode='LCH'
        )

        trainset = mz.dataloader.Dataset(
            data_pack=train_pack_processed,
            mode='pair',
            num_dup=5,
            num_neg=10,
            callbacks=[histgram_callback]
        )
        testset = mz.dataloader.Dataset(
            data_pack=test_pack_processed,
            callbacks=[histgram_callback]
        )