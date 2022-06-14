import os, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.CII as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')
    datasets = config['test']['dataset']

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = [getattr(module_loss, _loss) for _loss in config['loss']]
    # loss_fn = config.init_ftn('loss', module_loss)
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    resume = Path(config.resume)
    test_dir =  resume.parents[1]

    assert os.path.exists(test_dir), "{} does not exist.".format(test_dir)
    logger.info('Testing on experiment: {} ...'.format(test_dir))

    assert os.path.exists(resume), "Checkpoint {} does not exist.".format(resume)
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    # state_dict = checkpoint
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    # for k in state_dict.keys():
    #     print(k)
    model.load_state_dict(state_dict)
    
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    for dataset in datasets:
        logger.info('Testing on dataset: {} ...'.format(dataset))
        # setup data_loader instances
        data_loader = getattr(module_data, config['data_loader']['type'])(
            data_dir=os.path.join('./data', dataset, 'Imgs'),
            data_list=os.path.join('./data', dataset, 'test.lst'),
            batch_size=1,
            image_size=config['data_loader']['args']['image_size'],
            shuffle=False,
            validation_split=0.0,
            training=False,
            num_workers=config['data_loader']['args']['num_workers']
        )

        results_dir = test_dir / 'results' / dataset
        results_dir.mkdir(parents=True, exist_ok=True)

        total_loss = 0.0
        total_metrics = torch.zeros(len(metric_fns))

        with torch.no_grad():
            for i, (data, target, image_name, image_size) in enumerate(tqdm(data_loader)):
                data, target = data.to(device).float(), target.to(device).float()
                output = model(data)

                # computing loss, metrics on test set
                loss = 0
                for _loss_fn in loss_fn:
                    loss += _loss_fn(output, target)
                # loss = loss_fn(output, target)
                batch_size = data.shape[0]
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target, device=device) * batch_size

                for output_, image_name_, image_size_ in zip(output, image_name, image_size):
                    if output_.shape[-2:] != image_size_:
                        output_ = F.interpolate(output_.unsqueeze(0), tuple(image_size_.int().tolist()), mode='bilinear', align_corners=True)
                    output_ = np.squeeze(torch.sigmoid(output_).cpu().data.numpy())

                    Image.fromarray(output_ * 255).convert("L").save(results_dir / (image_name_[:-4] + '.png'))

        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
