import numpy as np
import logging
import os

logger = logging.getLogger(__name__)


def get_elmo(model_path, use_cuda=True):
    from elmoformanylangs.gen_elmo import Model
    from elmoformanylangs.modules.embedding_layer import EmbeddingLayer
    import codecs
    import json
    from os.path import dirname, join as pjoin
    config = json.load(open(pjoin(dirname(__file__), "../vendor/cnn_50_100_512_4096_sample.json")))
    # For the model trained with character-based word encoder.
    if config['token_embedder']['char_dim'] > 0:
        char_lexicon = {}
        with codecs.open(os.path.join(model_path, 'char.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                char_lexicon[token] = int(i)
        char_emb_layer = EmbeddingLayer(config['token_embedder']['char_dim'], char_lexicon, fix_emb=False, embs=None)
        logging.info('char embedding size: ' + str(len(char_emb_layer.word2id)))
    else:
        char_lexicon = None
        char_emb_layer = None

    # For the model trained with word form word encoder.
    if config['token_embedder']['word_dim'] > 0:
        word_lexicon = {}
        with codecs.open(os.path.join(model_path, 'word.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                word_lexicon[token] = int(i)
        word_emb_layer = EmbeddingLayer(config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=None)
        logging.info('word embedding size: ' + str(len(word_emb_layer.word2id)))
    else:
        word_lexicon = None
        word_emb_layer = None

    # instantiate the model
    model = Model(config, word_emb_layer, char_emb_layer, use_cuda)

    if use_cuda:
        model.cuda()

    logging.info(str(model))
    model.load_model(model_path)

    # I'll just keep this here...
    model.config = config
    model.char_lexicon = char_lexicon
    model.word_lexicon = word_lexicon
    model.use_cuda = use_cuda

    return model


def read_sent(sent, max_chars=None):
    dataset = []
    textset = []
    data = ['<bos>']
    text = []
    for token in sent:
        text.append(token)
        if max_chars is not None and len(token) + 2 > max_chars:
            token = token[:max_chars - 2]
        data.append(token)
    data.append('<eos>')
    dataset.append(data)
    textset.append(text)
    return dataset, textset


def get_layer(data, layer):
    if layer == -2:
        return np.r_[np.expand_dims(np.average(data, axis=0), 0), data]
    elif layer == -1:
        return np.average(data, axis=0)
    else:
        return data[layer]


def embed_sentence(model, sent, output_layer=-1):
    from elmoformanylangs.gen_elmo import create_one_batch
    config = model.config
    if config['token_embedder']['name'].lower() == 'cnn':
        batch_x, batch_text = read_sent(sent, config['token_embedder']['max_characters_per_token'])
    else:
        batch_x, batch_text = read_sent(sent)

    batch_w, batch_c, batch_lens, batch_masks = create_one_batch(
        batch_x,
        model.word_lexicon,
        model.char_lexicon,
        config,
        sort=False,
        use_cuda=model.use_cuda
    )

    output = model.forward(batch_w, batch_c, batch_masks)
    for i, text in enumerate(batch_text):
        if config['encoder']['name'].lower() == 'lstm':
            data = output[i, 1:batch_lens[i]-1, :].data
            if model.use_cuda:
                data = data.cpu()
            data = data.numpy()
        elif config['encoder']['name'].lower() == 'elmo':
            data = output[:, i, 1:batch_lens[i]-1, :].data
            if model.use_cuda:
                data = data.cpu()
            data = data.numpy()
    return get_layer(data, output_layer)


def read_sents(sents, max_chars=None):
    dataset = []
    textset = []
    for sent in sents:
        data = ['<bos>']
        text = []
        for token in sent:
            text.append(token)
            if max_chars is not None and len(token) + 2 > max_chars:
                token = token[:max_chars - 2]
            data.append(token)
        data.append('<eos>')
        dataset.append(data)
        textset.append(text)
    return dataset, textset


def create_batches(x, batch_size, word2id, char2id, config, use_cuda=False):
    from elmoformanylangs.gen_elmo import create_one_batch
    batches_w, batches_c, batches_lens, batches_masks = [], [], [], []
    size = batch_size
    nbatch = (len(x) - 1) // size + 1
    for i in range(nbatch):
        start_id, end_id = i * size, (i + 1) * size
        bw, bc, blens, bmasks = create_one_batch(
            x[start_id: end_id], word2id, char2id, config,
            sort=False, use_cuda=use_cuda)
        batches_w.append(bw)
        batches_c.append(bc)
        batches_lens.append(blens)
        batches_masks.append(bmasks)

    return batches_w, batches_c, batches_lens, batches_masks


def embed_sentences(model, sents, output_layer=-1, batch_size=64):
    config = model.config
    if config['token_embedder']['name'].lower() == 'cnn':
        batch_x, batch_text = read_sents(sents, config['token_embedder']['max_characters_per_token'])
    else:
        batch_x, batch_text = read_sents(sents)
    batch_w, batch_c, batch_lens, batch_masks = create_batches(
        batch_x, batch_size, model.word_lexicon, model.char_lexicon, config, use_cuda=model.use_cuda)
    res = []
    for w, c, lens, masks in zip(batch_w, batch_c, batch_lens, batch_masks):
        output = model.forward(w, c, masks)
        for i in range(len(lens)):
            if config['encoder']['name'].lower() == 'lstm':
                data = output[i, 1:lens[i]-1, :].data
            elif config['encoder']['name'].lower() == 'elmo':
                data = output[:, i, 1:lens[i]-1, :].data
            if model.use_cuda:
                data = data.cpu()
            data = data.numpy()
            res.append(get_layer(data, output_layer))
    return res
