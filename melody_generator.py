"""" 
RNN-GAN networks to generate a piece of melody
parameters:

to run:

"""
import os

import tensorflow as tf
import rnn_gan_graph
import melody_utils

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("datadir", None,
                           "Directory to save and load midi music files.")
tf.app.flags.DEFINE_string("traindir", None,
                           "Directory to save checkpoints and gnuplot files.")
tf.app.flags.DEFINE_integer("select_validation_percentage", 20,
                            "Select random percentage of data as validation set.")
tf.app.flags.DEFINE_integer("select_test_percentage", 20,
                            "Select random percentage of data as test set.")
tf.app.flags.DEFINE_integer("num_max_epochs", 4000,
                            "Select random percentage of data as test set.")


def melody_train(graph, loader, config, summary_frequency=2):
    global_step = graph.get_collection('global_step')[0]
    input_melody = graph.get_collection('input_melody')[0]
    loss_d = graph.get_collection('loss_d')[0]
    loss_g = graph.get_collection('loss_g')[0]
    clip_d_op = graph.get_collection('clip_d_op')[0]
    train_g_op = graph.get_collection('train_g_op')[0]
    output_melody = graph.get_collection('output_melody')[0]
    g_pre_train_op = graph.get_collection('g_pre_train_op')[0]
    pre_loss_g = graph.get_collection('pre_loss_g')[0]

    sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.traindir, save_model_secs=120, global_step=global_step)

    with sv.managed_session() as session:
        global_step_ = session.run(global_step)
        tf.logging.info('Starting training loop...')
        while global_step_ < FLAGS.num_max_epochs:
            if sv.should_stop():
                break
            d_iters = 4
            for _ in range(d_iters):
                batch_songs = loader.get_batch(config.batch_size, config.song_length)
                iter_loss_d, clip_d_op_ = session.run([loss_d, clip_d_op], {input_melody: batch_songs})

            batch_songs = loader.get_batch(config.batch_size, config.song_length)
            g_op, global_step_, g_loss, d_loss, output_melody_ = session.run(
                [train_g_op, global_step, loss_g, loss_d, output_melody], {input_melody: batch_songs})

            if global_step_ % 20 == 0 and global_step_ > 0:
                print('Global_step: %d    loss_d: %f    loss_g: %f', global_step_, d_loss, g_loss)
                print(output_melody_[0][0: 10])

            # if global_step_ % 5 == 0 and global_step_ != 0:
            #     outputs = session.run([output_melody])
            #      # sv.saver.save(session, sv.save_path, global_step=sv.global_step)
            #     print ('Global_step is actually increased')
            #     print outputs

            # # Pre-train
            # mark, batch_songs = loader.get_batch(config.batch_size, config.song_length)
            # if global_step_ > 100 and global_step_ % 20 == 0:
            #     output_melody_, pre_loss_g_, global_step_, _ = session.run([output_melody, pre_loss_g, global_step,
            #                                                              g_pre_train_op], {input_melody: batch_songs})
            #     print('Epoch: %d  pre_loss: %f', global_step_, pre_loss_g_)
            #     print(output_melody_[0][0:10])
            # else:
            #     pre_loss_g_, global_step_, _ = session.run([pre_loss_g, global_step, g_pre_train_op],
            #                                                {input_melody: batch_songs})
            #     print('Epoch: %d  pre_loss: %f', global_step_, pre_loss_g_)






def main(_):
    print(tf.__version__)
    if not FLAGS.datadir or not os.path.exists(FLAGS.datadir):
        raise ValueError("Must set --datadir to midi music dir.")
    if not FLAGS.traindir:
        raise ValueError("Must set --traindir to dir where I can save model and plots.")
    print('Begin Create Graph....')

    if not os.path.exists(FLAGS.traindir):
        try:
            os.makedirs(FLAGS.traindir)
        except:
            raise IOError
    print('Train dir: %s', FLAGS.traindir)

    melody_param = MelodyParam()
    config = rnn_gan_graph.RnnGanConfig(melody_param=melody_param)
    graph = rnn_gan_graph.build_graph(config)
    loader = melody_utils.MusicDataLoader(FLAGS.datadir, FLAGS.select_validation_percentage,
                                          FLAGS.select_test_percentage, config)


    melody_train(graph, loader, config)


class MelodyParam:
    def __init__(self, ticks_max=240, ticks_min=0, length_max=240, length_min=15, pitch_max=84, pitch_min=40,
                 velocity_max=127, velocity_min=100):
        self.ticks_max = ticks_max
        self.ticks_min = ticks_min
        self.length_max = length_max
        self.length_min = length_min
        self.pitch_max = pitch_max
        self.pitch_min = pitch_min
        self.velocity_max = velocity_max
        self.velocity_min = velocity_min

        self.nor_ticks = (ticks_max - ticks_min) / 15 + 1
        self.nor_length = (length_max - length_min) / 15 + 1
        self.nor_pitch = pitch_max - pitch_min
        self.nor_velocity = velocity_max - velocity_min

        self.total_length = self.nor_ticks + self.nor_length + self.nor_pitch + self.nor_velocity

        self.ticks_weight = self.nor_ticks / float(self.total_length)
        self.length_weight = self.nor_length / float(self.total_length)
        self.pitch_weight = self.nor_pitch / float(self.total_length)
        self.velocity_weight = self.nor_velocity / float(self.total_length)



if __name__ == '__main__':
    tf.app.run()