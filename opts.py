import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file",
                            default = 'dataset/BenchmarkingZeroShot/topic/train_pu_half_v0.txt',
                            type = str,
                            help = "The train set.")
    
    parser.add_argument("--dev_file",
                            default ='dataset/BenchmarkingZeroShot/topic/dev.txt',
                            type = str,
                            help = "The validation set.")
    
    parser.add_argument("--test_file",
                            default = 'dataset/BenchmarkingZeroShot/topic/test.txt',
                            type = str,
                            help = "The test set.")

    ## Other parameters

    parser.add_argument("--word_emb_dim",
                        default=300,
                        type=int,
                        help="The pre-trained word embeddings dimension.")

    parser.add_argument("--seq_max_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                                "Sequences longer than this will be truncated, and sequences shorter \n"
                                "than this will be padded.")

    parser.add_argument("--topic_max_length",
                        default=3,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                                "Sequences longer than this will be truncated, and sequences shorter \n"
                                "than this will be padded.")
    parser.add_argument("--min_word_freq",
                        default=1,
                        type=int,
                        help="# threshold for word frequency.")
    
    parser.add_argument("--fine_tune_word_embeddings",
                        action='store_true',
                        help="# fine-tune pre-trained word embeddings?")
    
    parser.add_argument("--expand_vocab",
                        action='store_true',
                        help="Whether to expand word vocab from pretrained word embeddings.")                  
    
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for eval.")
    
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--epochs",
                        default=50,
                        type=int,
                        help="Total number of training epochs to perform.")
    
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    args = parser.parse_args()

    return args