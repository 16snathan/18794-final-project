#adapted version of code found at https://medium.com/@fanzongshaoxing/model-flops-measurement-in-tensorflow-a84084bbb3b5
import tensorflow as tf


def load_pb(pb):
    with tf.io.gfile.GFile(pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        #first arg of shape is 1, but in reality the placeholder tensor first dim is 'None'
        c = tf.constant(False, dtype=float, shape=(1,299,299,3),name="ConstantInp")
        tf.import_graph_def(graph_def, name='',input_map={"Placeholder:0": c})
        return graph

def estimate_flops(pb_model):
    graph = load_pb(pb_model)
    with graph.as_default():
        flops = tf.compat.v1.profiler.profile(graph, options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
        print('Model {} needs {} FLOPS'.format(pb_model, flops.total_float_ops))

model = "baseline_tf_inception_v3_graph.pb"
estimate_flops(model)
