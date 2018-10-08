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
    if output_layer == -1:
        payload = np.average(data, axis=0)
    else:
        payload = data[output_layer]
    return payload
