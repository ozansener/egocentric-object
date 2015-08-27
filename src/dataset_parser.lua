require 'torch'
require 'paths'
require 'lfs'
require 'image'

DIR_SEP = "/" -- Directory seperator

local module = {}

function parse_folder(folder_name)
   local image_files = {}
   local image_id = 0
   local gt_file
   for file_name in lfs.dir(folder_name) do
      if file_name~="." and file_name~=".." then
         local full_path = folder_name .. DIR_SEP .. file_name
         local mode = lfs.attributes(full_path, "mode")
         if mode =="file" then
            if string.find(full_path,'png') then
               image_files[image_id]=full_path
               image_id = image_id+1
            else
               gt_file = full_path
            end
         end
      end
   end
   return image_files, gt_file
end


function read_gt_file(file_name)
   local gt = {}
   for line in io.lines(file_name) do
      image_name, digit_id = string.match(line,"([^:]*):([^:]*)")
      image_id = string.match(image_name, 'image([0-9]*).png$')
      gt[tonumber(image_id)]=tonumber(digit_id)
   end
   return gt
end


function read_and_save(folder_name, output_name)
   local image_files, gt_file = parse_folder(folder_name)
   local labels = read_gt_file(gt_file)

   local dataset = {}
   dataset.data = torch.Tensor(#labels+1,1,32,32) -- +1 is for 0/1 index
   dataset.labels = torch.Tensor(#labels+1)


    for id, file_name in pairs(image_files) do
       file_id = string.match(file_name, ".*image([0-9]+).png$")
       image_id = tonumber(file_id) + 1

       local image_data = image.load(file_name,1,'byte')

       dataset.data[{image_id,{},{},{}}] = image_data
       dataset.labels[image_id] = labels[image_id - 1] + 1
    end

    torch.save(output_name,dataset, 'ascii')
end

module.read_and_save = read_and_save
return module




