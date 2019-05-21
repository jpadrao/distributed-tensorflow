# Parameter Server implementation

## server

parameter server implementation

### flags

**sync_mode** - synchronous or asynchronous server (default synchronous)

**n_nodes** - number of workers training the model

**address** - server address (default 127.0.0.1), port fixed at 6000

**n_cores** - number of cores on the parameter server node (default 1)

**output_file** - file to dump the logs (default None -> terminal)


**example**

```
python3 server.py --sync_mode sync --n_nodes 30
```

## client

worker node implementation

### flags

**batch_size** - size of each batch

**n_epochs** - number of epochs (default 1)

**n_nodes** - number of worker nodes

**address** - server address (default 127.0.0.1), port fixed at 6000

**output_file** - file to dump the logs (default None -> terminal)

**split_dataset** - if True splits the dataset for each node (default 1 (True))

**example**

```
python3 client.py --n_nodes 30 --batch_size 32
```