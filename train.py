#!/usr/bin/env python

# Modified Horovod MNIST example

import os
import sys
import time

import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import graphics
from utils import ResultLogger

learn = tf.contrib.learn

# Surpress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _print(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs)


def init_visualizations(hps, model, logdir):

    def sample_batch(y, eps):
        n_batch = hps.local_batch_train
        xs = []
        for i in range(int(np.ceil(len(eps) / n_batch))):
            xs.append(model.sample(
                y[i*n_batch:i*n_batch + n_batch], eps[i*n_batch:i*n_batch + n_batch]))
        return np.concatenate(xs)

    def draw_samples(epoch, save=True, out_times=False):
        if hvd.rank() != 0:
            return

        rows = 10 if hps.image_size <= 64 else 4
        cols = rows
        n_batch = rows*cols
        y = np.asarray([_y % hps.n_y for _y in (
            list(range(cols)) * rows)], dtype='int32')

        # temperatures = [0., .25, .5, .626, .75, .875, 1.] #previously
        temperatures = [0., .25, .5, .6, .7, .8, .9, 1.]

        times = [time.time()]
        x_samples = []
        x_samples.append(sample_batch(y, [.0]*n_batch))
        times += [time.time()]
        x_samples.append(sample_batch(y, [.25]*n_batch))
        times += [time.time()]
        x_samples.append(sample_batch(y, [.5]*n_batch))
        times += [time.time()]
        x_samples.append(sample_batch(y, [.6]*n_batch))
        times += [time.time()]
        x_samples.append(sample_batch(y, [.7]*n_batch))
        times += [time.time()]
        x_samples.append(sample_batch(y, [.8]*n_batch))
        times += [time.time()]
        x_samples.append(sample_batch(y, [.9] * n_batch))
        times += [time.time()]
        x_samples.append(sample_batch(y, [1.]*n_batch))
        times += [time.time()]

        times = [a - b for a, b in zip(times[1:], times[:-1])]

        print('Sample times: {} mean: {} std: {}'.format(
            times, np.mean(times), np.std(times, ddof=1)))
        # previously: 0, .25, .5, .625, .75, .875, 1.

        if save:
            for i in range(len(x_samples)):

                x_sample = np.reshape(
                    x_samples[i], (n_batch, hps.image_size, hps.image_size, hps.channels))
                graphics.save_raster(x_sample, logdir +
                                    'epoch_{}_sample_{}.png'.format(epoch, i))
        if out_times:
            return np.array(times)

    return draw_samples
# ===
# Code for getting data
# ===
def get_data(hps, sess):

    if hps.problem == 'space':
        hps.n_train = 5000

    if hps.channels == -1:
        hps.channels = 1 if hps.problem == 'histo' else 3

    if hps.image_size == -1:
        hps.image_size = {'mnist': 32, 'histo': 28, 'space': 64, 'cifar10': 32, 'imagenet-oord': 64,
                          'imagenet-sample':32,
                          'imagenet': 256, 'celeba': 256, 'lsun_realnvp': 64, 'lsun': 256}[hps.problem]
    if hps.n_test == -1:
        hps.n_test = {'mnist': 10000, 'histo': 2000, 'space': 5000, 'cifar10': 10000, 'imagenet-oord': 50000, 'imagenet': 50000,
                      'imagenet-sample':10000,'celeba': 3000, 'lsun_realnvp': 300*hvd.size(), 'lsun': 300*hvd.size()}[hps.problem]
    hps.n_y = {'mnist': 10, 'histo': 1, 'space': 1, 'cifar10': 10, 'imagenet-oord': 1000, 'imagenet-sample':1000,
               'imagenet': 1000, 'celeba': 1, 'lsun_realnvp': 1, 'lsun': 1}[hps.problem]
    if hps.data_dir == "":
        hps.data_dir = {'mnist': None, 'histo': None, 'space': None, 'cifar10': None, 'imagenet-sample':None,'imagenet-oord': '/local/ehoogebo/imagenet-oord-tfr', 'imagenet': '/mnt/host/imagenet-tfr',
                        'celeba': '/mnt/host/celeba-reshard-tfr', 'lsun_realnvp': '/mnt/host/lsun_realnvp', 'lsun': '/mnt/host/lsun'}[hps.problem]

    if hps.problem == 'lsun_realnvp':
        hps.rnd_crop = True
    else:
        hps.rnd_crop = False

    if hps.category:
        hps.data_dir += ('/%s' % hps.category)

    # Use anchor_size to rescale batch size based on image_size
    s = hps.anchor_size
    hps.local_batch_train = hps.n_batch_train * \
        s * s // (hps.image_size * hps.image_size)
    hps.local_batch_test = {64: 50, 32: 25, 28: 25, 16: 10, 8: 5, 4: 2, 2: 2, 1: 1}[
        hps.local_batch_train]  # round down to closest divisor of 50

    hps.local_batch_init = hps.n_batch_init * \
        s * s // (hps.image_size * hps.image_size)

    print("Rank {} Batch sizes Train {} Test {} Init {}".format(
        hvd.rank(), hps.local_batch_train, hps.local_batch_test, hps.local_batch_init))

    if hps.problem in ['imagenet-oord', 'imagenet', 'celeba', 'lsun_realnvp', 'lsun']:
        hps.direct_iterator = True
        import data_loaders.get_data as v
        train_iterator, test_iterator, data_init = \
            v.get_data(sess, hps.data_dir, hvd.size(), hvd.rank(), hps.pmap, hps.fmap, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size, hps.rnd_crop)

    elif hps.problem in ['mnist', 'cifar10', 'histo', 'space', 'imagenet-sample']:
        hps.direct_iterator = False
        import data_loaders.get_mnist_cifar as v
        if hps.problem == 'imagenet-sample':
            if hps.image_size != 32:
                raise Exception
            problem = 'cifar10'
        else:
            problem = hps.problem
        train_iterator, test_iterator, data_init = \
            v.get_data(problem, hvd.size(), hvd.rank(), hps.dal, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size)

    else:
        raise Exception()

    return train_iterator, test_iterator, data_init


def process_results(results):
    stats = ['loss', 'bits_x', 'bits_y', 'pred_loss']
    assert len(stats) == results.shape[0]
    res_dict = {}
    for i in range(len(stats)):
        res_dict[stats[i]] = "{:.4f}".format(results[i])
    return res_dict


def count_params():
    total_parameters = 0
    conv1x1_parameters = 0
    fourier_params = 0
    emerging_params = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = [dim.value for dim in variable.get_shape()]

        if 'model/' in variable.name:
            if 'emerging/' in variable.name:
                from conv2d.conv2d import get_conv_square_ar_mask
                if len(shape) > 2:
                    emerging_params += np.sum(get_conv_square_ar_mask(*shape))
                else:
                    emerging_params += np.prod(shape)

            elif 'invconv/' in variable.name:
                conv1x1_parameters += np.prod(shape)
            elif 'fourier/' in variable.name:
                fourier_params += np.prod(shape)
            else:
                total_parameters += np.prod(shape)

    print('1x1', 100*conv1x1_parameters / total_parameters)
    print('emerging', 100*emerging_params / total_parameters)
    print('fourier', 100*fourier_params / total_parameters)
    print(total_parameters + emerging_params + conv1x1_parameters +
          fourier_params)


def main(hps):

    # Initialize Horovod.
    hvd.init()

    # Create tensorflow session
    sess = tensorflow_session()

    # Download and load dataset.
    tf.set_random_seed(hvd.rank() + hvd.size() * hps.seed)
    np.random.seed(hvd.rank() + hvd.size() * hps.seed)

    # Get data and set train_its and valid_its
    train_iterator, test_iterator, data_init = get_data(hps, sess)
    hps.train_its, hps.test_its, hps.full_test_its = get_its(hps)

    # Create log dir
    logdir = os.path.abspath(hps.logdir) + "/"
    if hvd.rank() == 0:
        if not os.path.exists(logdir):
            os.mkdir(logdir)

    if hps.inference or hps.restore :
        hps.restore_path = logdir + "model_best_loss.ckpt"

    # Create model
    import model
    model = model.model(sess, hps, train_iterator, test_iterator, data_init)

    if hps.graph:
        writer = tf.summary.FileWriter(logdir+'graph', sess.graph)

    count_params()

    # Initialize visualization functions
    visualise = init_visualizations(hps, model, logdir)

    if hps.inference:
        infer(sess, model, hps, test_iterator)
    elif hps.sample:
        sample(sess, model, hps, visualise)
    else:
        # Perform training
        train(sess, model, hps, logdir, visualise)
    
def sample(sess, model, hps, visualise):
    print('Testing...')
    times = []
    n_test = 2
    for i in range(n_test):
        print("Sample {}/{}".format(i+1,n_test))
        t = visualise(0,save=False, out_times=True)
        times.append(t)
    filename = "logs/times_{}_p{}_c{}.npy".format(
        hps.problem,
        hps.flow_permutation,
        hps.flow_coupling
    )
    np.save(filename,np.concatenate(times, axis=0))


def infer(sess, model, hps, iterator, reconstruction=False):
    # Example of using model in inference mode. Load saved model using hps.restore_path
    # Can provide x, y from files instead of dataset iterator
    # If model is uncondtional, always pass y = np.zeros([bs], dtype=np.int32)
    print('Testing...')
    if hps.direct_iterator:
        iterator = iterator.get_next()

    xs = []
    zs = []
    forward_times = []
    if reconstruction:
        recons = []
        reverse_times = []
    for it in range(hps.full_test_its):
        if hps.direct_iterator:
            # replace with x, y, attr if you're getting CelebA attributes, also modify get_data
            x, y = sess.run(iterator)
        else:
            x, y = iterator()

        t = time.perf_counter()
        z = model.encode(x, y)
        tt = time.perf_counter()
        forward_time = tt - t

        if reconstruction:
            t = time.perf_counter()
            recon = model.decode(y, z)
            tt = time.perf_counter()
            reverse_time = tt - t
            reverse_times.append(reverse_time)
            recons.append(recon)

        forward_times.append(forward_time)
        xs.append(x)
        zs.append(z)

    x = np.concatenate(xs, axis=0)
    z = np.concatenate(zs, axis=0)
    forward_times = np.array(forward_times)
    np.save('logs/x.npy', x)
    np.save('logs/z.npy', z)
    np.save('logs/forward_times.npy', forward_times)

    if reconstruction:
        recon = np.concatenate(recons, axis=0)
        reverse_times = np.array(reverse_times)
        np.save('logs/recon.npy', recon)
        np.save('logs/reverse_times.npy', reverse_times)

    return zs



def train(sess, model, hps, logdir, visualise):
    _print(hps)
    _print('Starting training. Logging to', logdir)
    _print('epoch n_processed n_images ips dtrain dtest dsample dtot train_results test_results msg')

    # Train
    sess.graph.finalize()
    n_processed = 0
    n_images = 0
    train_time = 0.0
    test_loss_best = 999999

    if hvd.rank() == 0:
        train_logger = ResultLogger(logdir + "train.txt", **hps.__dict__)
        test_logger = ResultLogger(logdir + "test.txt", **hps.__dict__)

    tcurr = time.time()
    for epoch in range(1, hps.epochs):

        t = time.time()

        train_results = []
        for it in range(hps.train_its):

            # Set learning rate, linearly annealed from 0 in the first hps.epochs_warmup epochs.
            lr = hps.lr * min(1., n_processed /
                              (hps.n_train * hps.epochs_warmup))

            # Run a training step synchronously.
            _t = time.time()
            train_results += [model.train(lr)]
            if hps.verbose and hvd.rank() == 0:
                _print(n_processed, time.time()-_t, train_results[-1])
                sys.stdout.flush()

            # Images seen wrt anchor resolution
            n_processed += hvd.size() * hps.n_batch_train
            # Actual images seen at current resolution
            n_images += hvd.size() * hps.local_batch_train

        train_results = np.mean(np.asarray(train_results), axis=0)

        dtrain = time.time() - t
        ips = (hps.train_its * hvd.size() * hps.local_batch_train) / dtrain
        train_time += dtrain

        if hvd.rank() == 0:
            train_logger.log(epoch=epoch, n_processed=n_processed, n_images=n_images, train_time=int(
                train_time), **process_results(train_results))

        if epoch < 10 or (epoch < 50 and epoch % 10 == 0) or epoch % hps.epochs_full_valid == 0:
            test_results = []
            msg = ''

            t = time.time()
            # model.polyak_swap()

            if epoch % hps.epochs_full_valid == 0:
                # Full validation run
                for it in range(hps.full_test_its):
                    test_results += [model.test()]
                test_results = np.mean(np.asarray(test_results), axis=0)

                if hvd.rank() == 0:
                    test_logger.log(epoch=epoch, n_processed=n_processed,
                                    n_images=n_images, **process_results(test_results))

                    # Save checkpoint
                    if test_results[0] < test_loss_best:
                        test_loss_best = test_results[0]
                        model.save(logdir+"model_best_loss.ckpt")
                        msg += ' *'

            dtest = time.time() - t

            # Sample
            t = time.time()
            if epoch == 10 or epoch % hps.epochs_full_sample == 0:
                visualise(epoch)
            dsample = time.time() - t

            if hvd.rank() == 0:
                dcurr = time.time() - tcurr
                tcurr = time.time()
                _print(epoch, n_processed, n_images, "{:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(
                    ips, dtrain, dtest, dsample, dcurr), train_results, test_results, msg)

            # model.polyak_swap()

    if hvd.rank() == 0:
        _print("Finished!")

# Get number of training and validation iterations
def get_its(hps):
    # These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
    train_its = int(np.ceil(hps.n_train / (hps.n_batch_train * hvd.size())))
    test_its = int(np.ceil(hps.n_test / (hps.n_batch_train * hvd.size())))
    train_epoch = train_its * hps.n_batch_train * hvd.size()

    # Do a full validation run
    if hvd.rank() == 0:
        print(hps.n_test, hps.local_batch_test, hvd.size())
    assert hps.n_test % (hps.local_batch_test * hvd.size()) == 0
    full_test_its = hps.n_test // (hps.local_batch_test * hvd.size())

    if hvd.rank() == 0:
        print("Train epoch size: " + str(train_epoch))
    return train_its, test_its, full_test_its


'''
Create tensorflow session with horovod
'''
def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    return sess


if __name__ == "__main__":

    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--restore_path", type=str, default='',
                        help="Location of checkpoint to restore")
    parser.add_argument("--inference", action="store_true",
                        help="Use in inference mode")
    parser.add_argument("--sample", action="store_true",
                        help="Use in sample mode to mesure sampling time")
    parser.add_argument("--restore", action="store_true",
                        help="Restore the model as it was saved")
    parser.add_argument("--graph", action="store_true",
                        help="Outputs the computational graph")
    parser.add_argument("--logdir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='cifar10',
                        help="Problem (mnist/cifar10/imagenet")
    parser.add_argument("--category", type=str,
                        default='', help="LSUN category")
    parser.add_argument("--data_dir", type=str, default='',
                        help="Location of data")
    parser.add_argument("--dal", type=int, default=1,
                        help="Data augmentation level: 0=None, 1=Standard, 2=Extra")

    # New dataloader params
    parser.add_argument("--fmap", type=int, default=1,
                        help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16,
                        help="# Threads for parallel map")

    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int, default=-
                        1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=64, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=50, help="Minibatch size")
    parser.add_argument("--n_batch_init", type=int, default=256,
                        help="Minibatch size for data-dependent init")
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=1000000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int,
                        default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=50, help="Epochs between valid")
    parser.add_argument("--gradient_checkpointing", type=int,
                        default=1, help="Use memory saving gradients")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=-1, help="Image size")
    parser.add_argument("--anchor_size", type=int, default=32,
                        help="Anchor size for deciding batch size")
    parser.add_argument("--channels", type=int, default=-1,
                        help="Number of channels in input ")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=32,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=3,
                        help="Number of levels")

    # 1x1 Decomposition
    parser.add_argument("--decomposition", type=str, default='',
                        help='decomposition of 1x1 conv')

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=1,
                        help="minibatch size for sample")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=50, help="Epochs between full scale sample")

    # Ablation
    parser.add_argument("--learnprior", action="store_true",
                        help="Learn (hierarchical) prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=2,
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=emerging conv, 7=inv conv (ours)")
    parser.add_argument("--flow_coupling", type=int, default=0,
                        help="Coupling type: 0=additive, 1=affine, 2=quad (ours)")

    hps = parser.parse_args()  # So error if typo
    main(hps)
