import kfp.dsl as dsl
import kfp.compiler as compiler
import sys


def train(volume_op):
    return dsl.ContainerOp(
        name="train",
        image="anbn1/anbn-kube",
        command=["python", "run_mnist.py"],
        arguments=["--train", "/mnt/trained_model.h5"],
        pvolumes={"/mnt": volume_op.volume},
        file_outputs = { "model": "/mnt/trained_model.h5" }
    )


def test(volume_op):
    return dsl.ContainerOp(
        name="test",
        image="anbn1/anbn-kube",
        command=["python", "run_mnist.py"],
        arguments=["--test", "/mnt/trained_model.h5"],
        pvolumes={"/mnt": volume_op.volume},
    )


def pipeline_volume(size="1Gi"):
    return dsl.VolumeOp(
        name="pipeline volume",
        resource_name="pipeline-pvc",
        modes=["ReadWriteOnce"],
        size=size,
    )


@dsl.pipeline(name="my_mnist_pipeline", description="")
def create_pipeline():
    volume_op = pipeline_volume()

    train_step = train(volume_op)
    test_step = test(volume_op)

    test_step.after(train_step)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)

    filename = sys.argv[1]
    compiler.Compiler().compile(create_pipeline, filename)
