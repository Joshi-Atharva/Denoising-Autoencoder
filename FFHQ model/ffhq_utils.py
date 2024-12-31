import numpy as np
import matplotlib.pyplot as plt

def PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def imshow_side_by_side(input, target, prediction):
    npimg_input = input.numpy()
    npimg_target = target.numpy()
    npimg_prediction = prediction.detach().numpy().squeeze(0)
    print(npimg_prediction.shape)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(np.transpose(npimg_input, (1, 2, 0)), cmap = 'gray')
    axs[1].imshow(np.transpose(npimg_target, (1, 2, 0)), cmap = 'gray')
    axs[2].imshow(np.transpose(npimg_prediction, (1, 2, 0)), cmap = 'gray')
    plt.show()

def predict(idx, model, device, test_set):
    model.eval()
    input, target = test_set[idx]
    input = input.to(device)
    target = target.to(device)
    output = model(input.unsqueeze(0))
    input = input.to('cpu')
    target = target.to('cpu')
    output = output.to('cpu')
    imshow_side_by_side(input, target, output)
    return output, target

def test(model, device, test_set):
    model.eval()
    accuracy_aggregator = 0
    for idx in range(len(test_set)):
        output, target = predict(idx, model, device, test_set)
        accuracy = PSNR(output.detach().cpu().numpy(), target.detach().cpu().numpy())
        print(f'Accuracy: {accuracy:.0f} dB')
        accuracy_aggregator += accuracy/len(test_set)
    print(f'Test Accuracy: {accuracy_aggregator:.0f} dB')
    return accuracy_aggregator
