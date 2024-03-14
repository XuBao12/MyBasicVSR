import matplotlib.pyplot as plt


def draw_loss(train_loss, args):
    save_file = args.log_dir

    fig = plt.figure()
    losses = [loss for loss in train_loss]
    x_train = range(len(losses))
    plt.title("losses of training")
    plt.plot(x_train, losses)
    fig.savefig(f"{save_file}/train_loss.png")


def draw_valrst(val_results, args):
    max_epoch = args.epochs
    save_file = args.log_dir

    x_val = [x for x in range(max_epoch) if (x + 1) % args.val_interval == 0]
    # psnr
    fig = plt.figure()
    val_psnr = [val_result["psnr"] for val_result in val_results]
    plt.title("psnr of validation")
    plt.plot(x_val, val_psnr, label="val_psnr")
    plt.legend()
    fig.savefig(f"{save_file}/psnr.png")

    # ssim
    fig = plt.figure()
    val_ssim = [val_result["ssim"] for val_result in val_results]
    plt.title("ssim of validation")
    plt.plot(x_val, val_ssim, label="val_ssim")
    plt.legend()
    fig.savefig(f"{args.log_dir}/ssim.png")
