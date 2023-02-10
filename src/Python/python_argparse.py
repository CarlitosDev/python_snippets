# Use argparse (pip3 install argparse)
import argparse
parser = argparse.ArgumentParser(
	description='Scripts reads Adform data stored in AWS S3 and transforms it to Parquet '
				'format and re-uploads it to S3. *** NOTE: for tailored for Trackingpoint ***')

parser.add_argument('adform_type', action='store', choices=['Trackingpoint', 'Impression', 'Clicks'],
					help='Adform type.')

parser.add_argument('download_path', action='store', help='path to download S3 files from source bucket.')