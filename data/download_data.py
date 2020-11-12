import os
import zipfile
import urllib
import urllib.request
import ssl

def download_dataset(source_url, target_dir, target_file):
    global downloaded
    downloaded = 0
    def show_progress(count, block_size, total_size):
        global downloaded
        downloaded += block_size
        print('downloading ... %d%%' % round(((downloaded*100.0) / total_size)))

    print('downloading ... ')
    urllib.request.urlretrieve(source_url, filename=target_file, reporthook=show_progress)
    print('downloading ... done')

    print('extracting ...')
    with zipfile.ZipFile(target_file, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    os.remove(target_file)
    print('extracting ... done')


if __name__ == '__main__':
    source_url = 'http://profs.scienze.univr.it/~marin/instant/data.zip'
    target_dir = os.path.dirname(os.path.abspath(__file__))
    target_file = os.path.join(target_dir, 'data')
    download_dataset(source_url,  target_dir, target_file)