import shutil
import tarfile
from boxsdk import OAuth2, Client
from pathlib import Path


def download_box_file(box_file, local_path: Path):
    with local_path.open('wb') as output_file:
        box_file.download_to(output_file)


def download_box_directory(box_directory, local_directory_path: Path):
    local_directory_path.mkdir(parents=True, exist_ok=True)
    for item in box_directory.get_items():
        item_path = local_directory_path / item.name
        if item.type == 'folder':
            item_path.mkdir(parents=True, exist_ok=True)
            download_box_directory(item, item_path)
        elif item.type == 'file':
            download_box_file(item, item_path)


def download_dataset() -> None:
    public_link = 'https://lsu.app.box.com/s/qx440yp9ekzrhaevtfu7ksnfgh2jhc29'  # Replace with your actual public link
    oauth2 = OAuth2(
        client_id='',
        client_secret='',
        access_token=''
    )

    client = Client(oauth2)

    box_directory = client.get_shared_item(public_link)

    local_directory = Path('data/gulls_orbital_motion_download')
    download_box_directory(box_directory, local_directory)

    print("File downloaded successfully.")


def extract_dataset() -> None:
    archive_root_directory = Path('data/gulls_orbital_motion')
    temporary_directory = Path('data/temporary')
    temporary_directory.mkdir(parents=True, exist_ok=True)
    destination_directory = Path('data/full_roman_simulated_microlensing')
    destination_directory.mkdir(parents=True, exist_ok=True)
    files_extracted = 0
    for archive in archive_root_directory.glob('OMPLLD_croin_cassan_*_*_lc.tar.gz'):
        print(archive.name)
        shutil.rmtree(temporary_directory)
        temporary_directory.mkdir(parents=True)
        with tarfile.open(archive, 'r:gz') as tar:
            tar.extractall(path=temporary_directory)
        for light_curve_path in temporary_directory.glob('**/*.det.lc'):
            light_curve_path: Path
            light_curve_path.rename(destination_directory.joinpath(light_curve_path.name))
            files_extracted += 1
    print(f'Files extracted: {files_extracted}')


if __name__ == '__main__':
    extract_dataset()
