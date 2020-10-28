import kfp.dsl as dsl

def train():
    container_op = dsl.ContainerOp(
     name='train',
     image='anbn1/anbn-kube',
     command=['python', 'run_mnist.py'],
     arguments=['--train', '/trained_model.h5'],
     file_outputs={ "model": '/trained_model.h5' }
    )
    return container_op

def test(train_step):
    container_op = dsl.ContainerOp(
     name='test',
     image='anbn1/anbn-kube',
     command=['python', 'run_mnist.py'],
     arguments=['--test', train_step.outputs['model']]
    )
    return container_op

@dsl.pipeline(name='my_mnist_pipeline', description='')
def create_pipeline():
    train_step = train()
    test_step = test(train_step)
    test_step.after(train_step)

if __name__ == '__main__':
  import kfp.compiler as compiler
  import sys
  if len(sys.argv) != 2:
    sys.exit(-1)

  filename = sys.argv[1]
  compiler.Compiler().compile(create_pipeline, filename)
