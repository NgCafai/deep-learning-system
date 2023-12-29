import sys
sys.path.append("./tests/hw4")
sys.path.append("./python")

from test_nd_backend import *
from test_cifar_ptb_data import *
from test_conv import *
from test_sequence_models import *
from needle import backend_ndarray as nd


def train_cifar10():
    import sys
    sys.path.append('./python')
    sys.path.append('./apps')
    import needle as ndl
    from models import ResNet9
    from simple_ml import train_cifar10, evaluate_cifar10

    device = ndl.cpu()
    dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    dataloader = ndl.data.DataLoader( \
        dataset=dataset,
        batch_size=128,
        shuffle=True, )
    model = ResNet9(device=device, dtype="float32")
    train_cifar10(model, dataloader, n_epochs=2, optimizer=ndl.optim.Adam,
                  lr=0.001, weight_decay=0.001, device=device)
    evaluate_cifar10(model, dataloader)


def train_language_model():
    import needle as ndl
    sys.path.append('./apps')
    from models import LanguageModel
    from simple_ml import train_ptb, evaluate_ptb

    device = ndl.cpu_numpy()
    corpus = ndl.data.Corpus("data/ptb")
    train_data = ndl.data.batchify(corpus.train, batch_size=16, device=device, dtype="float32")
    model = LanguageModel(30, len(corpus.dictionary), hidden_size=10, num_layers=2, seq_model='rnn', device=device)
    train_ptb(model, train_data, seq_len=1, n_epochs=1, device=device)
    evaluate_ptb(model, train_data, seq_len=40, device=device)


if __name__ == "__main__":
    """
    Part 1
    """
    # test_stack((5, 5), 0, 2, nd.cpu())
    # test_stack_backward((5, 5), 0, 2, nd.cpu())

    # test_matmul(16, 16, 16, nd.cpu())
    # test_relu((5, 5), nd.cpu())
    # test_tanh_backward((5, 5), nd.cpu())


    """
    Part 2
    """
    # test_cifar10_dataset(True)


    """
    Part 3
    """
    # test_pad_forward({"shape": (10, 32, 32, 8), "padding": ( (0, 0), (2, 2), (2, 2), (0, 0) )}, nd.cpu())
    # test_flip_forward({"shape": (10, 5), "axes": (0,)}, nd.cpu())
    # test_dilate_forward(nd.cpu())
    # test_op_conv((3, 16, 16, 8), (3, 3, 8, 16), 1, 2, False, nd.cpu())
    # test_op_conv((3, 16, 16, 8), (3, 3, 8, 16), 2, 1, True, nd.cpu())

    # test_init_kaiming_uniform(nd.cpu())
    # test_nn_conv_forward(4, 8, 16, 3, 1, nd.cpu())
    # test_nn_conv_backward(4, 1, 1, 3, 1, nd.cpu())
    # test_resnet9(nd.cpu())
    # test_train_cifar10(nd.cpu())

    train_cifar10()

    """
    Part 4
    """
    # test_rnn_cell(1, 1, 1, False, False, 'relu', nd.cpu())
    # test_lstm_cell(1, 1, 1, False, False, nd.cpu())
    # test_lstm(13, 1, 1, 1, 1, True, True, nd.cpu())

    """
    Part 6
    """
    # test_language_model_implementation(1, 1, 1, 1, 1, True, 1, 'rnn', nd.cpu())

    """
    Part 7
    """
    # train_language_model()