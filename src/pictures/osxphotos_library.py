'''osxphotos_library.py


git clone https://github.com/RhetTbull/osxphotos.git
cd osxphotos
python3 setup.py install
'''


import osxphotos

library_path = '/Volumes/CarlosPictures/ExternalDrive.photoslibrary'

photosdb = osxphotos.PhotosDB(library_path)




# print(photosdb.album_names)
# print(photosdb.albums_as_dict)

print(photosdb.keywords_as_dict)



print(photosdb.persons)
persons_dict = photosdb.persons_as_dict


photos = photosdb.photos()

photos_list = [p for p in photos]

len(photos_list)

one_pic = photos_list[3123]
type(one_pic)

search_info = one_pic.search_info.all 
filepath = one_pic.path
one_pic.face_info
one_pic.location
one_pic.original_filename
picture_labels = one_pic.labels

import json
pic_info = json.loads(one_pic.json())
pic_info.keys()
import carlos_utils.file_utils as fu
fu.printJSON(pic_info)


'''

one_pic.adjustments            one_pic.date_trashed           one_pic.hasadjustments         one_pic.labels                 one_pic.path_derivatives       one_pic.search_info_normalized
one_pic.album_info             one_pic.description            one_pic.hdr                    one_pic.labels_normalized      one_pic.path_edited            one_pic.SearchInfo(           
one_pic.albums                 one_pic.detected_text(         one_pic.height                 one_pic.likes                  one_pic.path_edited_live_photo one_pic.selfie                
one_pic.asdict(                one_pic.duplicates             one_pic.hidden                 one_pic.live_photo             one_pic.path_live_photo        one_pic.shared                
one_pic.burst                  one_pic.exif_info              one_pic.import_info            one_pic.location               one_pic.path_raw               one_pic.slow_mo               
one_pic.burst_album_info       one_pic.ExifInfo(              one_pic.incloud                one_pic.moment                 one_pic.person_info            one_pic.time_lapse            
one_pic.burst_albums           one_pic.exiftool               one_pic.intrash                one_pic.orientation            one_pic.persons                one_pic.title                 
one_pic.burst_default_pick     one_pic.export(                one_pic.iscloudasset           one_pic.original_filename      one_pic.place                  one_pic.tzoffset              
one_pic.burst_key              one_pic.export2(               one_pic.ismissing              one_pic.original_filesize      one_pic.portrait               one_pic.uti                   
one_pic.burst_photos           one_pic.ExportResults(         one_pic.ismovie                one_pic.original_height        one_pic.raw_original           one_pic.uti_edited            
one_pic.burst_selected         one_pic.external_edit          one_pic.isphoto                one_pic.original_orientation   one_pic.render_template(       one_pic.uti_original          
one_pic.comments               one_pic.face_info              one_pic.israw                  one_pic.original_width         one_pic.score                  one_pic.uti_raw               
one_pic.date                   one_pic.favorite               one_pic.isreference            one_pic.owner                  one_pic.ScoreInfo(             one_pic.uuid                  
one_pic.date_added             one_pic.filename               one_pic.json(                  one_pic.panorama               one_pic.screenshot             one_pic.visible               
one_pic.date_modified          one_pic.has_raw                one_pic.keywords               one_pic.path                   one_pic.search_info            one_pic.width    

'''