#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:47:32 2018

@author: shinong
"""
def create_inception_graph():
  """"Creates a graph from saved GraphDef file and returns a Graph object.
  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  with tf.Session() as sess:
    model_filename = os.path.join(
        FLAGS.model_dir, 'classify_image_graph_def.pb')
    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def main(_):
  # Setup the directory we'll write summaries to for TensorBoard
  if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
  tf.gfile.MakeDirs(FLAGS.summaries_dir)

  # Set up the pre-trained graph.
  maybe_download_and_extract()
  graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
      create_inception_graph())

  # Look at the folder structure, and create lists of all the images.
  image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                   FLAGS.validation_percentage)

  if len(image_lists.keys()) == 0:
      print('Folder containing training images has not been found inside {} directory. \n'
            'Put all the training images into '
            'one folder inside {} directory and delete everything else inside the {} directory.'
            .format(FLAGS.image_dir, FLAGS.image_dir, FLAGS.image_dir))
      return -1

  if len(image_lists.keys()) > 1:
      print('More than one folder found inside {} directory. \n'
            'In order to prevent validation issues, put all the training images into '
            'one folder inside {} directory and delete everything else inside the {} directory.'
            .format(FLAGS.image_dir, FLAGS.image_dir, FLAGS.image_dir))
      return -1

  if not os.path.isfile(ALL_LABELS_FILE):
      print('File {} containing all possible labels (= classes) does not exist.\n'
            'Create it in project root and put each possible label on new line, '
            'it is exactly the same as creating an image_label file for image '
            'that is in all the possible classes.'.format(ALL_LABELS_FILE))
      return -1

  with open(ALL_LABELS_FILE) as f:
      labels = f.read().splitlines()
  class_count = len(labels)

  if class_count == 0:
    print('No valid labels inside file {} that should contain all possible labels (= classes).'.format(ALL_LABELS_FILE))
    return -1
  if class_count == 1:
    print('Only one valid label found inside {} - multiple classes are needed for classification.'.format(ALL_LABELS_FILE))
    return -1

  # See if the command-line flags mean we're applying any distortions.
  do_distort_images = should_distort_images(
      FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
      FLAGS.random_brightness)
  sess = tf.Session()

  if do_distort_images:
    # We will be applying distortions, so setup the operations we'll need.
    distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)
  else:
    # We'll make sure we've calculated the 'bottleneck' image summaries and
    # cached them on disk.
    cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor)

  # Add the new layer that we'll be training.
  (train_step, cross_entropy, bottleneck_input, ground_truth_input,
   final_tensor) = add_final_training_ops(class_count,
                                          FLAGS.final_tensor_name,
                                          bottleneck_tensor)

  # Create the operations we need to evaluate the accuracy of our new layer.
  evaluation_step = add_evaluation_step(final_tensor, ground_truth_input)

  # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                        sess.graph)
  validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

  # Set up all our weights to their initial default values.
  init = tf.initialize_all_variables()
  sess.run(init)

  # Run the training for as many cycles as requested on the command line.
  for i in range(FLAGS.how_many_training_steps):
    # Get a batch of input bottleneck values, either calculated fresh every time
    # with distortions applied, or from the cache stored on disk.
    if do_distort_images:
      train_bottlenecks, train_ground_truth = get_random_distorted_bottlenecks(
          sess, image_lists, FLAGS.train_batch_size, 'training',
          FLAGS.image_dir, distorted_jpeg_data_tensor,
          distorted_image_tensor, resized_image_tensor, bottleneck_tensor, labels)
    else:
      train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
          sess, image_lists, FLAGS.train_batch_size, 'training',
          FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
          bottleneck_tensor, labels)
    # Feed the bottlenecks and ground truth into the graph, and run a training
    # step. Capture training summaries for TensorBoard with the `merged` op.
    train_summary, _ = sess.run([merged, train_step],
             feed_dict={bottleneck_input: train_bottlenecks,
                        ground_truth_input: train_ground_truth})
    train_writer.add_summary(train_summary, i)

    # Every so often, print out how well the graph is training.
    is_last_step = (i + 1 == FLAGS.how_many_training_steps)
    if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
      train_accuracy, cross_entropy_value = sess.run(
          [evaluation_step, cross_entropy],
          feed_dict={bottleneck_input: train_bottlenecks,
                     ground_truth_input: train_ground_truth})
      print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                      train_accuracy * 100))
      print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                 cross_entropy_value))
      validation_bottlenecks, validation_ground_truth = (
          get_random_cached_bottlenecks(
              sess, image_lists, FLAGS.validation_batch_size, 'validation',
              FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
              bottleneck_tensor, labels))
      # Run a validation step and capture training summaries for TensorBoard
      # with the `merged` op.
      validation_summary, validation_accuracy = sess.run(
          [merged, evaluation_step],
          feed_dict={bottleneck_input: validation_bottlenecks,
                     ground_truth_input: validation_ground_truth})
      validation_writer.add_summary(validation_summary, i)
      print('%s: Step %d: Validation accuracy = %.1f%%' %
            (datetime.now(), i, validation_accuracy * 100))

  # We've completed all our training, so run a final test evaluation on
  # some new images we haven't used before.
  test_bottlenecks, test_ground_truth = get_random_cached_bottlenecks(
      sess, image_lists, FLAGS.test_batch_size, 'testing',
      FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
      bottleneck_tensor, labels)
  test_accuracy = sess.run(
      evaluation_step,
      feed_dict={bottleneck_input: test_bottlenecks,
                 ground_truth_input: test_ground_truth})
  print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

  # Write out the trained graph and labels with the weights stored as constants.
  output_graph_def = graph_util.convert_variables_to_constants(
      sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
  with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
    f.write(output_graph_def.SerializeToString())
  with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
    f.write('\n'.join(image_lists.keys()) + '\n')


if __name__ == '__main__':
    tf.app.run()