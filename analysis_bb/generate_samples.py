# train_set = BouncingBallDataLoader('dataset/bouncing_ball/200/train', seen_len = 50)
# train_loader = torch.utils.data.DataLoader(
#                     dataset=train_set, 
#                     batch_size=1, 
#                     shuffle=True)

# data, target = next(iter(train_loader))
# print(data.shape, target.shape)

def plot_seq(seq, filepath, filename = None, seq_list = None):
    os.makedirs(filepath, exist_ok = True)
    if seq_list is None: 
        seq_list = torch.arange(0, seq.size(1))

    for batch_item, i in enumerate(seq): 
        i = i[seq_list]
        frames = torchvision.utils.make_grid(i,i.size(0))

        if filename is None: 
            plt.imsave(filepath + f"seq_{batch_item}.jpeg",
                frames.cpu().permute(1, 2, 0).numpy())
        else: 
            plt.imsave(filepath + filename + f"{batch_item}.jpeg",
                frames.cpu().permute(1, 2, 0).numpy())

    return 

# plot_seq(target, filepath, "target", seq_list = torch.arange(0, 150, 5))

# ### Save as numpy array 
# data = data.detach().cpu().numpy()
# target = target.detach().cpu().numpy()

# data = data[0]
# target = target[0]
# np.savez("analysis_bb/BouncingBall_100/samples/input", data)
# np.savez("analysis_bb/BouncingBall_100/samples/target", target)


# data_64 = np.zeros((50, 1, 64, 64))
# for t, frame in enumerate(data): 
#     frame = np.transpose(frame, (1, 2, 0))
#     frame = cv2.resize(frame, dsize = (64, 64))
#     data_64[t,0,:,:] = frame 

# data_64 = (data_64 - data_64.min()) / (data_64.max() - data_64.min())

# target_64 = np.zeros((150, 1, 64, 64))
# for t, frame in enumerate(target): 
#     frame = np.transpose(frame, (1, 2, 0))
#     frame = cv2.resize(frame, dsize = (64, 64))
#     target_64[t,0,:,:] = frame 

# target_64 = (target_64 - target_64.min()) / (target_64.max() - target_64.min())

# np.savez("analysis_bb/BouncingBall_100/samples/input_big", data_64)
# np.savez("analysis_bb/BouncingBall_100/samples/target_big", target_64)
