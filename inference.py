import os
import sys

import tensorflow as tf
import numpy as np
import math

import tarfile

import utils
import nn
import argparse
import preprocessing

model_dir = './data/imagenet'
data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None

def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--textstring', help='the text you want to generate', default='Generating text', type=str)  
    parser.add_argument('--writersource', help="path of the image of the desired writer, (e.g. './assets/image.png'   \
                                                will use random from ./assets if unspecified", default=None)
    parser.add_argument('--name', help="path for generated image (e.g. './assets/sample.png'), \
                                             will not be saved if unspecified", default=None)
    parser.add_argument('--diffmode', help="what kind of y_t-1 prediction to use, use 'standard' for  \
                                            Eq 9 in paper, will default to prediction in Eq 12", default='new', type=str)
    parser.add_argument('--show', help="whether to show the sample (popup from matplotlib)", default=False, type=bool)
    parser.add_argument('--weights', help='the path of the loaded weights', default='./weights/model_weights.h5', type=str)
    parser.add_argument('--seqlen', help='number of timesteps in generated sequence, default 16 * length of text', default=None, type=int)
    parser.add_argument('--num_attlayers', help='number of attentional layers at lowest resolution, \
                                                 only change this if loaded model was trained with that hyperparameter', default=2, type=int)
    parser.add_argument('--channels', help='number of channels at lowest resolution, only change \
                                                 this if loaded model was trained with that hyperparameter', default=128, type=int)
    
    args = parser.parse_args()
    timesteps = len(args.textstring) * 16 if args.seqlen is None else args.seqlen
    timesteps = timesteps - (timesteps%8) + 8 
    #must be divisible by 8 due to downsampling layers

    if args.writersource is None:
        assetdir = os.listdir('./assets')
        sourcename = './assets/' + assetdir[np.random.randint(0, len(assetdir))]
    else: 
        sourcename = args.writersource
 
    
    tokenizer = utils.Tokenizer()
    beta_set = utils.get_beta_set()
    

    C1 = args.channels
    C2 = C1 * 3//2
    C3 = C1 * 2
    style_extractor = nn.StyleExtractor()
    model = nn.DiffusionWriter(num_layers=args.num_attlayers, c1=C1, c2=C2, c3=C3)
    
    _stroke = tf.random.normal([1, 400, 2])
    _text = tf.random.uniform([1, 40], dtype=tf.int32, maxval=50)
    _noise = tf.random.uniform([1, 1])
    _style_vector = tf.random.normal([1, 14, 1280])
    _ = model(_stroke, _text, _noise, _style_vector)
    #we have to call the model on input first
    model.load_weights(args.weights)

    writer_img = tf.expand_dims(preprocessing.read_img(sourcename, 96), 0)
    style_vector = style_extractor(writer_img)
    utils.run_batch_inference(model, beta_set, args.textstring, style_vector, 
                                tokenizer=tokenizer, time_steps=timesteps, diffusion_mode=args.diffmode, 
                                show_samples=args.show, path=args.name)


def get_inception_score(create_session, imgs, splits=10, bs=100):
    init_inception(create_session)
    score = _get_inception_score(create_session, imgs, splits, bs)
    tf.reset_default_graph()
    return score


def _get_inception_score(create_session, images, splits=10, bs=100):
    assert (type(images) == list)
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    assert (np.max(images[0]) > 10)
    assert (np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    with create_session() as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        for i in range(n_batches):
            #sys.stdout.write(".")
            #sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {'ExpandDims:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)


def init_inception(create_session):
    global softmax
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
            
        print()
    tarfile.open(filepath, 'r:gz').extractall(model_dir)
    with tf.gfile.FastGFile(os.path.join(model_dir, 'classify_img.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

        model, preprocess = clip.load("ViT-B/32")
        

    def init_clip(screate_session, images, splits=10, bs=100, self, clip, preprocess):
         super().__init__()
            self.clip=clip
            self.preprocess=preprocess
                
        def forward(self, x):
            batch_features=[]
            with tf.no_grad():
                
                x = self.transform(x[0])
                image = self.preprocess(x).unsqueeze(0)
                image_features = model.encode_image(image)
                batch_features.append(image_features)
                return tf.cat(batch_features)

    with create_session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
          
                o.__dict__['_shape_val'] = tf.TensorShape(new_shape)
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)


if __name__ == '__main__':
    main()