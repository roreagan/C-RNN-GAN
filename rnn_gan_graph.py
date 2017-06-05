"""Provide function to build an RNN-GAN model's graph. """

import tensorflow as tf

def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell,
                  state_is_tuple=True,
                  reuse=False):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the RNN.
    dropout_keep_prob: The float probability to keep the output of any given sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
    state_is_tuple: A boolean specifying whether to use tuple of hidden matrix
        and cell matrix as a state instead of a concatenated matrix.

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units, state_is_tuple=state_is_tuple, reuse=reuse)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length, state_is_tuple=state_is_tuple)

  return cell

def data_type():
    return tf.float32  # this can be replaced with tf.float32

def discriminator(config, inputs, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    fw_cell_d = make_rnn_cell(config.d_rnn_layers, reuse=reuse)
    bw_cell_d = make_rnn_cell(config.d_rnn_layers, reuse=reuse)
    init_state_fw_d = fw_cell_d.zero_state(config.batch_size, data_type())
    init_state_bw_d = bw_cell_d.zero_state(config.batch_size, data_type())
    outputs, final_state_fw, final_state_bw = \
        tf.contrib.rnn.static_bidirectional_rnn(fw_cell_d, bw_cell_d, inputs,
                                                initial_state_fw=init_state_fw_d,
                                                initial_state_bw=init_state_bw_d, scope='bidirection_rnn')
    decisions = tf.sigmoid(tf.contrib.layers.fully_connected(outputs, 1, scope='decision'))
    decisions = tf.transpose(decisions, perm=[1, 0, 2])
    decision = tf.reduce_mean(decisions, reduction_indices=[1, 2])
    return decision

def build_graph(config):
    batch_size = config.batch_size
    song_length = config.song_length
    num_song_features = config.num_song_features
    with tf.Graph().as_default() as graph:
        # input_melody is a seed with the shape of [batch_size, note_length, num_song_features] to generate a melody.
        input_melody = tf.placeholder(dtype=tf.int32, shape=[batch_size, song_length, num_song_features])

        global_step = tf.Variable(0, trainable=False, name='global_step')

        tf.add_to_collection('input_melody', input_melody)
        tf.add_to_collection('global_step', global_step)

        with tf.variable_scope('G') as scopeG:
            cell_g = make_rnn_cell(config.g_rnn_layers)  # set to [300, 300]
            init_state_g = cell_g.zero_state(batch_size, data_type())

            # Train
            # generate a random note as a seed with the shape of [batch_size, num_song_features] to
            # generate a piece of melody with the length of notes_length
            random_ticks = tf.random_uniform(shape=[batch_size, config.ticks_length], minval=0,
                                             maxval=config.ticks_length-1, dtype=data_type())
            random_length = tf.random_uniform(shape=[batch_size, config.length_length], minval=0,
                                             maxval=config.length_length - 1, dtype=data_type())
            random_pitch = tf.random_uniform(shape=[batch_size, config.pitch_length], minval=0,
                                             maxval=config.pitch_length - 1, dtype=data_type())
            random_velocity = tf.random_uniform(shape=[batch_size, config.velocity_length], minval=0,
                                             maxval=config.velocity_length - 1, dtype=data_type())
            # random_rnn_inputs' shape is [batch_size, num_song_features]
            random_rnn_inputs = tf.to_float(tf.concat([random_ticks, random_length, random_pitch, random_velocity], 1))

            output_melody = []
            output_melody_probability = []
            generated_note = random_rnn_inputs
            state_g = init_state_g

            for i in range(song_length):
                if i > 0:
                    scopeG.reuse_variables()
                inputs = tf.contrib.layers.fully_connected(generated_note, config.g_rnn_layers[0],
                                                           scope='note_to_input')
                outputs, state_g = cell_g(inputs, state_g)

                pitch_logits_flat = tf.contrib.layers.fully_connected(outputs, config.pitch_length,
                                                                      scope='output_to_pitch')
                ticks_logits_flat = tf.contrib.layers.fully_connected(outputs, config.ticks_length,
                                                                      scope='output_to_ticks')
                velocity_logits_flat = tf.contrib.layers.fully_connected(outputs, config.velocity_length,
                                                                         scope='output_to_velocity')
                length_logits_flat = tf.contrib.layers.fully_connected(outputs, config.length_length,
                                                                       scope='output_to_length')
                output_melody_probability.append([ticks_logits_flat, length_logits_flat,
                                                           pitch_logits_flat, velocity_logits_flat])
                pitchs = [tf.arg_max(tf.squeeze(note), 0) for note in
                          tf.split(pitch_logits_flat, batch_size, 0)]
                ticks = [tf.arg_max(tf.squeeze(note), 0) for note in
                          tf.split(ticks_logits_flat, batch_size, 0)]
                velocitys = [tf.arg_max(tf.squeeze(note), 0) for note in
                          tf.split(velocity_logits_flat, batch_size, 0)]
                lengths = [tf.arg_max(tf.squeeze(note), 0) for note in
                          tf.split(length_logits_flat, batch_size, 0)]
                generated_note = tf.to_float(tf.stack([ticks, lengths, pitchs, velocitys], axis=1))
                output_melody.append(generated_note)

            # Pretraining
            # PreTraining
            # get softmax cross entropy of every song features
            # softmax for every features shows the probable value of one step
            labels_flat = tf.reshape(input_melody, shape=[-1, num_song_features])

            ticks_labels_flat = tf.squeeze(tf.slice(labels_flat, [-1, 0], [-1, 1]))
            length_labels_flat = tf.squeeze(tf.slice(labels_flat, [-1, 0], [-1, 1]))
            pitch_labels_flat = tf.squeeze(tf.slice(labels_flat, [-1, 0], [-1, 1]))
            velocity_labels_flat = tf.squeeze(tf.slice(labels_flat, [-1, 0], [-1, 1]))

            ticks_logits_flat_sum = [note[0] for note in output_melody_probability]
            length_logits_flat_sum = [note[1] for note in output_melody_probability]
            pitch_logits_flat_sum = [note[2] for note in output_melody_probability]
            velocity_logits_flat_sum = [note[3] for note in output_melody_probability]

            ticks_softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=ticks_labels_flat, logits=ticks_logits_flat_sum)
            length_softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=length_labels_flat, logits=length_logits_flat_sum)
            pitch_softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=pitch_labels_flat, logits=pitch_logits_flat_sum)
            velocity_softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=velocity_labels_flat, logits=velocity_logits_flat_sum)
            pre_loss_g = tf.reduce_sum(tf.add(tf.add(tf.add(ticks_softmax_cross_entropy,
                                                                length_softmax_cross_entropy),
                                                         pitch_softmax_cross_entropy),
                                                  velocity_softmax_cross_entropy)) / batch_size
            tf.add_to_collection('pre_loss_g', pre_loss_g)

            g_learning_rate = tf.train.exponential_decay(
                config.initial_g_learning_rate, global_step, config.decay_steps,
                config.decay_rate, staircase=True, name='learning_rate')

            # g_opt = tf.train.AdamOptimizer(g_learning_rate)
            g_opt = tf.train.GradientDescentOptimizer(g_learning_rate)
            g_params = [v for v in tf.trainable_variables() if v.name.startswith('G/')]
            g_gradients = tf.gradients(pre_loss_g, g_params)
            clipped_gradients, _ = tf.clip_by_global_norm(g_gradients, config.clip_norm)
            g_pre_train_op = g_opt.apply_gradients(zip(clipped_gradients, g_params), global_step)
            tf.add_to_collection('g_learning_rate', g_learning_rate)
            tf.add_to_collection('g_train_op', g_pre_train_op)

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = config.reg_constant
        reg_loss = reg_constant * sum(reg_losses)

        with tf.variable_scope('D') as scopeD:
            inputs_d = [tf.to_float(tf.squeeze(input_d)) for input_d in tf.split(input_melody, song_length, 1)]
            real_d = discriminator(config, inputs_d)
            generated_d = discriminator(config, output_melody, reuse=True)
            if config.wgan:
                loss_d = tf.reduce_mean(real_d) - tf.reduce_mean(generated_d)
                loss_g = tf.reduce_mean(generated_d)
            else:
                loss_d = tf.reduce_mean(-tf.log(tf.clip_by_value(real_d, 1e-1000000, 1.0)) \
                               - tf.log(1 - tf.clip_by_value(generated_d, 0.0, 1.0 - 1e-1000000)))
                loss_g = tf.reduce_mean(-tf.log(tf.clip_by_value(generated_d, 1e-1000000, 1.0)))

        d_params = [v for v in tf.trainable_variables() if v.name.startswith('D/')]
        g_params = [v for v in tf.trainable_variables() if v.name.startswith('G/')]

        loss_d = loss_d + reg_loss
        loss_g = loss_g + reg_loss
        optimizer_d = tf.train.RMSPropOptimizer(learning_rate=config.initial_learning_rate)
        optimizer_g = tf.train.RMSPropOptimizer(learning_rate=config.initial_learning_rate)

        d_grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss_d, d_params, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
            config.max_grad_norm)
        g_grads = tf.gradients(loss_g, g_params, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        train_d_op = optimizer_d.apply_gradients(zip(d_grads, d_params))
        clip_d_op = [w.assign(tf.clip_by_value(w, -config.clip_w_norm, config.clip_w_norm)) for w in d_params]
        train_g_op = optimizer_g.apply_gradients(zip(g_grads, g_params))
        tf.add_to_collection('loss_d', loss_d)
        tf.add_to_collection('loss_g', loss_g)
        tf.add_to_collection('train_d_op', train_d_op)
        tf.add_to_collection('clip_d_op', clip_d_op)
        tf.add_to_collection('train_g_op', train_g_op)










class RnnGanConfig:
    def __init__(self):
        self.batch_size = 10
        self.song_length = 10
        self.num_song_features = 4
        self.g_rnn_layers = [5, 5]
        self.d_rnn_layers = [5, 5]
        self.pitch_length = 50
        self.ticks_length = 20
        self.length_length = 30
        self.velocity_length = 50
        self.clip_norm = 5
        self.initial_g_learning_rate = 0.01
        self.initial_learning_rate = 0.0001
        self.decay_steps = 1000
        self.decay_rate = 0.95
        self.wgan = True
        self.reg_constant = 0.01  # choose a appropriate one
        self.max_grad_norm = 5
        self.clip_w_norm = 0.01


def main(_):
    config = RnnGanConfig()
    build_graph(config)


if __name__ == "__main__":
    tf.app.run()





