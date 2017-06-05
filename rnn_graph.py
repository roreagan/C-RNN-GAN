"""Provide function to build an RNN model's graph. """

import tensorflow as tf

def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell,
                  state_is_tuple=False):
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
    cell = base_cell(num_units, state_is_tuple=state_is_tuple)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length, state_is_tuple=state_is_tuple)

  return cell

def data_type():
    return tf.float16  # this can be replaced with tf.float32

def build_graph(config):
    batch_size = config.batch_size
    # song_length = config.song_length
    num_song_features = config.num_song_features
    with tf.Graph().as_default() as graph:
        # input_melody is a seed with the shape of [batch_size, note_length, num_song_features] to generate a step.
        # In train process, input_melody and the relative primer_melody are given.
        # In generate process, input_melody is randomly generated as the input of generator.
        input_melody, primer_melody = None, None
        primer_melody = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, num_song_features])
        input_melody = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, num_song_features])

        global_step = tf.Variable(0, trainable=False, name='global_step')

        tf.add_to_collection('input_melody', input_melody)
        tf.add_to_collection('primer_melody', primer_melody)
        tf.add_to_collection('global_step', global_step)

        with tf.variable_scope('G') as scopeG:
            cell_g = make_rnn_cell(config.g_rnn_layers)  # set to [300, 300]
            init_state_g = cell_g.zero_state(batch_size, data_type())
            inputs = tf.contrib.layers.linear(input_melody, config.g_rnn_layers[0])
            outputs, final_state_g = tf.nn.dynamic_rnn(
                cell_g, inputs, initial_state=init_state_g, parallel_iterations=1,
                swap_memory=True)
            outputs_flat = tf.reshape(outputs, [-1, cell_g.output_size])
            pitch_logits_flat = tf.contrib.layers.linear(outputs_flat, config.pitch_length)
            ticks_logits_flat = tf.contrib.layers.linear(outputs_flat, config.ticks_length)
            velocity_logits_flat = tf.contrib.layers.linear(outputs_flat, config.velocity_length)
            length_logits_flat = tf.contrib.layers.linear(outputs_flat, config.length_length)

            # PreTraining
            # get softmax cross entropy of every song features
            # softmax for every features shows the probable value of one step
            labels_flat = tf.reshape(primer_melody, shape=[-1, num_song_features])

            ticks_labels_flat = tf.reshape(tf.slice(labels_flat, [-1, 0], [-1, 1]), [-1])
            length_labels_flat = tf.reshape(tf.slice(labels_flat, [-1, 0], [-1, 1]), [-1])
            pitch_labels_flat = tf.reshape(tf.slice(labels_flat, [-1, 0], [-1, 1]), [-1])
            velocity_labels_flat = tf.reshape(tf.slice(labels_flat, [-1, 0], [-1, 1]), [-1])

            ticks_softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=ticks_labels_flat, logits=ticks_logits_flat)
            length_softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=length_labels_flat, logits=length_logits_flat)
            pitch_softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=pitch_labels_flat, logits=pitch_logits_flat)
            velocity_softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=velocity_labels_flat, logits=velocity_logits_flat)
            loss = tf.reduce_sum(tf.add(tf.add(tf.add(ticks_softmax_cross_entropy, length_softmax_cross_entropy),
                                               pitch_softmax_cross_entropy), velocity_softmax_cross_entropy))

            tf.add_to_collection('loss', loss)
            learning_rate = tf.train.exponential_decay(
                config.initial_learning_rate, global_step, config.decay_steps,
                config.decay_rate, staircase=True, name='learning_rate')

            opt = tf.train.AdamOptimizer(learning_rate)
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, config.clip_norm)
            train_op = opt.apply_gradients(zip(clipped_gradients, params), global_step)
            tf.add_to_collection('learning_rate', learning_rate)
            tf.add_to_collection('train_op', train_op)

            # Generate a step for generating. Returned softmaxs contain probabilities of every one-hot features

            temperature = tf.placeholder(data_type(), [])

            ticks_softmax_flat = tf.nn.softmax(
                tf.div(ticks_logits_flat, tf.fill([config.ticks_length], temperature)))
            ticks_softmax = tf.reshape(ticks_softmax_flat, [batch_size, -1, config.ticks_length])

            length_softmax_flat = tf.nn.softmax(
                tf.div(length_logits_flat, tf.fill([config.length_length], temperature)))
            length_softmax = tf.reshape(length_softmax_flat, [batch_size, -1, config.length_length])

            pitch_softmax_flat = tf.nn.softmax(
                tf.div(pitch_logits_flat, tf.fill([config.pitch_length], temperature)))
            pitch_softmax = tf.reshape(pitch_softmax_flat, [batch_size, -1, config.pitch_length])

            velocity_softmax_flat = tf.nn.softmax(
                tf.div(velocity_logits_flat, tf.fill([config.velocity_length], temperature)))
            velocity_softmax = tf.reshape(velocity_softmax_flat, [batch_size, -1, config.velocity_length])

            tf.add_to_collection('temperature', temperature)
            tf.add_to_collection('ticks_softmax', ticks_softmax)
            tf.add_to_collection('length_softmax', length_softmax)
            tf.add_to_collection('pitch_softmax', pitch_softmax)
            tf.add_to_collection('velocity_softmax', velocity_softmax)
            tf.add_to_collection('initial_state', init_state_g)
            tf.add_to_collection('final_state', final_state_g)










