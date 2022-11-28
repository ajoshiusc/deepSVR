def slice_generator(image, direction, slice_no, profile):
    slice_volume = torch.zeros(image.size())
    slice_width = profile.size(0)
    bottom = max(slice_no-int(profile.size(0)/2), 0)
    top = bottom + profile.size(0)
    slice_ind = torch.tensor(range(bottom, top))
    
    if direction == 0:
        new = torch.sum(torch.zeros(image.size()),2, keepdim = True)
        print(new.size())
        print(image[:,:,1,:,:].size())
        for i in slice_ind:
            new += image[:,:,i:i+1,:,:]*profile[i-bottom]
        slice_volume[:,:,slice_ind,:,:] = new.repeat(1,1,profile.size(0),1,1)
        
    elif direction == 1:
        new = torch.sum(torch.zeros(image.size()),3, keepdim = True)
        for i in slice_ind:
            new += image[:,:,:,i:i+1,:]*profile[i-bottom]
        slice_volume[:,:,:,slice_ind,:] = new.repeat(1,1,1,profile.size(0),1)
        
    elif direction == 2:
        new = torch.sum(torch.zeros(image.size()),4, keepdim = True)
        for i in slice_ind:
            new += image[:,:,:,:,i:i+1]*profile[i-bottom]
        slice_volume[:,:,:,:,slice_ind] = new.repeat(1,1,1,1,profile.size(0))
    
    return slice_volume