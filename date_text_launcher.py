import ast
import os
import sys
import argparse
import logging

import tensorflow as tf
from src.model.model import Model
from src import exp_config

tf.logging.set_verbosity(tf.logging.ERROR)


def as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


def process_args(args, defaults):

    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', dest="phase", type=str, default=defaults.PHASE, choices=['train', 'test'],
                        help=('Phase of experiment, can be either train or test, default=%s' % defaults.PHASE))

    parser.add_argument('--gpu-id', dest="gpu_id", type=int, default=defaults.GPU_ID)

    parser.add_argument('--channel', dest="channel", type=int, default=defaults.CHANNEL, choices=[1, 3],
                        help=('channel of input image, can be either 1 or 3, default=%s' % defaults.CHANNEL))

    parser.add_argument('--mean', dest='mean', type=as_list, default=defaults.MEAN,
                        help={'mean of training set'})

    parser.add_argument('--visualize', dest='visualize', action='store_true',
                        help=('Visualize attentions or not, default=%s' % defaults.VISUALIZE))
    parser.add_argument('--no-visualize', dest='visualize', action='store_false')
    parser.set_defaults(visualize=defaults.VISUALIZE)

    parser.add_argument('--use-gru', dest='use_gru', action='store_true')
    parser.set_defaults(use_gru=defaults.USE_GRU)

    parser.add_argument('--load-model', dest='load_model', action='store_true',
                        help=('Load model from model-dir or not, default=%s' % defaults.LOAD_MODEL))
    parser.add_argument('--no-load-model', dest='load_model', action='store_false')
    parser.set_defaults(load_model=defaults.LOAD_MODEL)

    # input and output for train or test
    parser.add_argument('--data-dir', dest="data_dir", type=str, default=defaults.DATA_DIR,
                        help=('root path of the files, default=%s' % defaults.DATA_DIR))

    parser.add_argument('--label-path', dest="label_path", type=str, default=defaults.LABEL_PATH,
                        help=('Path of file containing the path and labels, default=%s' % defaults.LABEL_PATH))

    parser.add_argument('--lexicon-file', dest="lexicon_file", type=str, default=defaults.LEXICON,
                        help=('Path of file containing lexicon, default=%s' % defaults.LEXICON))

    parser.add_argument('--model-dir', dest="model_dir", type=str, default=defaults.MODEL_DIR,
                        help=('The directory for saving and loading model (structure is not stored), default=%s'
                              % defaults.MODEL_DIR))

    parser.add_argument('--log-path', dest="log_path", type=str, default=defaults.LOG_PATH,
                        help=('Log file path, default=%s' % defaults.LOG_PATH))

    parser.add_argument('--output-dir', dest="output_dir",type=str, default=defaults.OUTPUT_DIR,
                        help=('Output directory, default=%s' % defaults.OUTPUT_DIR))

    parser.add_argument('--steps-per-checkpoint', dest="steps_per_checkpoint",
                        type=int, default=defaults.STEPS_PER_CHECKPOINT,
                        help=('Checkpoint for print perplexity and save model, default = %s'
                              % defaults.STEPS_PER_CHECKPOINT))

    # optimization
    parser.add_argument('--num-epoch', dest="num_epoch", type=int, default=defaults.NUM_EPOCH,
                        help=('Number of epochs, default = %s' % defaults.NUM_EPOCH))

    parser.add_argument('--batch-size', dest="batch_size", type=int, default=defaults.BATCH_SIZE,
                        help=('Batch size, default = %s' % defaults.BATCH_SIZE))

    parser.add_argument('--initial-learning-rate', dest="initial_learning_rate",
                        type=float, default=defaults.INITIAL_LEARNING_RATE,
                        help=('Initial learning rate, default = %s' % defaults.INITIAL_LEARNING_RATE))

    # network parameters
    parser.add_argument('--no-gradient_clipping', dest='clip_gradients', action='store_false',
                        help=('Do not perform gradient clipping, difault for clip_gradients is %s'
                              % defaults.CLIP_GRADIENTS))
    parser.set_defaults(clip_gradients=defaults.CLIP_GRADIENTS)

    parser.add_argument('--max_gradient_norm', dest="max_gradient_norm", type=int, default=defaults.MAX_GRADIENT_NORM,
                        help=('Clip gradients to this norm, default=%s' % defaults.MAX_GRADIENT_NORM))

    parser.add_argument('--target-embedding-size', dest="target_embedding_size",
                        type=int, default=defaults.TARGET_EMBEDDING_SIZE,
                        help=('Embedding dimension for each target, default=%s' % defaults.TARGET_EMBEDDING_SIZE))

    parser.add_argument('--target-vocab-size', dest="target_vocab_size", type=int, default=defaults.TARGET_VOCAB_SIZE,
                        help=('Target vocabulary size, default=%s' % defaults.TARGET_VOCAB_SIZE))

    parser.add_argument('--attn-num-hidden', dest="attn_num_hidden", type=int, default=defaults.ATTN_NUM_HIDDEN,
                        help=('hidden units in attention decoder cell, default=%s' % defaults.ATTN_NUM_HIDDEN))

    parser.add_argument('--attn-num-layers', dest="attn_num_layers", type=int, default=defaults.ATTN_NUM_LAYERS,
                        help=('number of hidden layers in attention decoder cell, default=%s'
                              % defaults.ATTN_NUM_LAYERS))

    parameters = parser.parse_args(args)

    return parameters


def main(args, defaults):

    parameters = process_args(args, defaults)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(parameters.gpu_id)

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
                        filename=parameters.log_path)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        model = Model(phase=parameters.phase,
                      gpu_id=parameters.gpu_id,
                      channel=parameters.channel,
                      mean=parameters.mean,
                      visualize=parameters.visualize,
                      use_gru=parameters.use_gru,
                      load_model=parameters.load_model,

                      data_dir=parameters.data_dir,
                      label_path=parameters.label_path,
                      lexicon_file=parameters.lexicon_file,
                      model_dir=parameters.model_dir,
                      output_dir=parameters.output_dir,
                      steps_per_checkpoint=parameters.steps_per_checkpoint,

                      num_epoch=parameters.num_epoch,
                      batch_size=parameters.batch_size,
                      initial_learning_rate=parameters.initial_learning_rate,

                      clip_gradients=parameters.clip_gradients,
                      max_gradient_norm=parameters.max_gradient_norm,
                      target_embedding_size=parameters.target_embedding_size,
                      attn_num_hidden=parameters.attn_num_hidden,
                      attn_num_layers=parameters.attn_num_layers,

                      valid_target_length=float('inf'),
                      session=sess)
        print('model init end, launch start...')
        model.launch()


def test():

    defaults = exp_config.ExpConfig

    parameters = dict()

    parameters['log_path'] = 'log.txt'
    parameters['phase'] = 'train'
    parameters['visualize'] = defaults.VISUALIZE
    parameters['data_path'] = 'train.txt'
    parameters['data_root_dir'] = '../data/date'
    parameters['lexicon_file'] = 'lexicon.txt'
    parameters['output_dir'] = defaults.OUTPUT_DIR
    parameters['batch_size'] = 4
    parameters['initial_learning_rate'] = 1.0
    parameters['num_epoch'] = 30
    parameters['steps_per_checkpoint'] = 200
    parameters['target_vocab_size'] = defaults.TARGET_VOCAB_SIZE
    parameters['model_dir'] = '../output'
    parameters['target_embedding_size'] = 10
    parameters['attn_num_hidden'] = defaults.ATTN_NUM_HIDDEN
    parameters['attn_num_layers'] = defaults.ATTN_NUM_LAYERS
    parameters['clip_gradients'] = defaults.CLIP_GRADIENTS
    parameters['max_gradient_norm'] = defaults.MAX_GRADIENT_NORM
    parameters['load_model'] = defaults.LOAD_MODEL
    parameters['gpu_id'] = defaults.GPU_ID
    parameters['use_gru'] = False

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters['log_path'])

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)) as sess:
        model = Model(
            phase=parameters['phase'],
            visualize=parameters['visualize'],
            data_path=parameters['data_path'],
            data_root_dir=parameters['data_root_dir'],
            output_dir=parameters['output_dir'],
            batch_size=parameters['batch_size'],
            initial_learning_rate=parameters['initial_learning_rate'],
            num_epoch=parameters['num_epoch'],
            steps_per_checkpoint=parameters['steps_per_checkpoint'],
            target_vocab_size=parameters['target_vocab_size'],
            model_dir=parameters['model_dir'],
            target_embedding_size=parameters['target_embedding_size'],
            attn_num_hidden=parameters['attn_num_hidden'],
            attn_num_layers=parameters['attn_num_layers'],
            clip_gradients=parameters['clip_gradients'],
            max_gradient_norm=parameters['max_gradient_norm'],
            load_model=parameters['load_model'],
            valid_target_length=float('inf'),
            gpu_id=parameters['gpu_id'],
            use_gru=parameters['use_gru'],
            session=sess)
        model.launch()


if __name__ == "__main__":
    main(sys.argv[1:], exp_config.ExpConfig)

