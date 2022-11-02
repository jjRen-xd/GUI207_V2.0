# -*- coding: utf-8 -*- #
import fractions
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gc

from typing import Any, List


class HookValues():
    """
        注册钩子，记录中间反传梯度
    """
    def __init__(self, layer) -> None:
        # register a hook to save values of activations and gradients
        self.activations = None
        self.gradients = None
        self.forward_hook = layer.register_forward_hook(self.hook_fn_act)
        self.backward_hook = layer.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


class BaseCAM():
    def __init__(self, imgs:np.ndarray, CLASSES: List[str]) -> None:
        for i in range(len(imgs)):
            imgs[i] -= np.min(imgs[i])
            imgs[i] /= np.max(imgs[i])
        self.imgs = imgs
        self.CLASSES = CLASSES
        self.bz, self.h, self.w, self.nc = imgs.shape

        self.activations = None
        self.gradients = None
        self.weights = None
        self.CAMs = None
        self.scaledCAMs = None


    def get_cam_weights(self, 
                activations:np.ndarray, 
                gradients:np.ndarray) -> np.ndarray:
        """
            TODO
        """
        raise Exception("Not Implemented")


    def get_cam_image(self,
                activations:np.ndarray, 
                gradients:np.ndarray,
                weights:np.ndarray) -> np.ndarray:
        cams = (weights[:, :, None, None] * activations).sum(axis=1)
        return cams


    def scale_image(self, imgs, target_size=None):
        result = []
        for img in imgs:
            if target_size is not None:
                img = cv2.resize(img, target_size, \
                    interpolation=cv2.INTER_CUBIC)
            img = img - np.min(img)
            if np.max(img) != 0:
                img = img / np.max(img)
            result.append(img)
        result = np.float32(result)

        return result


    def show_cam(self,
                 labelList: List[List[np.ndarray]],
                 imgList: List[np.ndarray] = None,
                 camList: List[np.ndarray] = None,
                 with_norm_in_bboxes=False):
        """Normalize the CAM to be in the range [0, 1] inside every bounding
        boxes, and zero outside of the bounding boxes."""
        if imgList is None:
            imgList = self.imgs
        else:
            imgList = self.scale_image(imgList, (self.w, self.h))
        if camList is None:
            camList = self.scaledCAMs
        else:
            camList = self.scale_image(camList, (self.w, self.h))
            
        result = []
        for image, grayscale_cam, labels in \
                                zip(imgList, camList, labelList):
            labels = t2n(labels)
            renormalized_cam = grayscale_cam

            cam_image_renormalized = self._overlay_cam_on_image(
                [image], [renormalized_cam])[0]

            result.append(cam_image_renormalized)
        return result


    def _overlay_cam_on_image(self, imgs: List[np.ndarray]=None,
                        cams: List[np.ndarray]=None,
                        layerName: str = ""):
        """ This function overlays the cam mask on the image as an heatmap.
        By default the heatmap is in BGR format.

        :param img: The base image in RGB or BGR format.
        :param cam: The cam mask.
        :returns: The default image with the cam overlay.
        """
        if imgs is None or cams is None:
            imgs = self.imgs
            cams = self.scaledCAMs
        result = []
        for sig, cam, CLASS in zip(imgs, cams, self.CLASSES):
            # CAM取值归一化
            sig_len, channel, _ = sig.shape
            cam = cam.T                      # (512, 1) -> (1, 512)
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)     
            sig_min, sig_max = np.min(sig), np.max(sig)    

            plt.figure(figsize=(18, 4))
            plt.title("Signal Type: "+CLASS+"    Model Layer: "+str(layerName))
            plt.xlabel('N')
            plt.ylabel("Value")
            plt.plot(sig[:, 0, 0]*(sig_len//10), color="c", label = 'hrrp')
            plt.legend(loc="upper right")
            plt.imshow(cam, cmap='jet', \
                extent=[0., sig_len, (sig_min-0.5)*(sig_len//10), (sig_max+0.5)*(sig_len//10)])     # jet, rainbow
            plt.colorbar(fraction=0.01)
            fig = plt.gcf()

            result.append(plt2cvMat(fig))

            plt.clf()
            plt.close()
            gc.collect()
        return result


    def __call__(self, activations:np.ndarray, gradients:np.ndarray) -> Any:
        self.activations = activations
        self.gradients = gradients
        assert self.activations is not None and self.gradients is not None

        # 计算CAM
        self.weights = self.get_cam_weights(self.activations, self.gradients)
        self.CAMs = self.get_cam_image(self.activations, self.gradients, \
                                        self.weights)
        
        # 删除小于0的值
        self.CAMs = np.maximum(self.CAMs, 0)    
        
        # 将CAM归一化，并将尺寸调整到图像大小target_size, 若不指定，则不调整
        self.scaledCAMs = self.scale_image(self.CAMs, (self.w, self.h))
        
        return self.scaledCAMs

    def __del__(self):
        # TODO
        pass

    def __enter__(self):
        # TODO
        pass

    def __exit__(self):
        # TODO
        pass


class GradCAM(BaseCAM):
    def __init__(self, imgs: np.ndarray, CLASSES: List[str]) -> None:
        super(GradCAM, self).__init__(imgs, CLASSES)

    def get_cam_weights(self, 
            activations:np.ndarray, 
            gradients:np.ndarray) -> np.ndarray:
        """
            TODO
        """
        weights = np.mean(gradients, axis=(2, 3))

        return weights


class GradCAMpp(BaseCAM):
    def __init__(self, imgs: np.ndarray, CLASSES: List[str]) -> None:
        super(GradCAMpp, self).__init__(imgs, CLASSES)

    def get_cam_weights(self, 
            activations:np.ndarray, 
            gradients:np.ndarray) -> np.ndarray:
        """
            TODO
        """
        grads_power_2 = gradients**2
        grads_power_3 = grads_power_2 * gradients
        # Equation 19 in https://arxiv.org/abs/1710.11063
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        aij = grads_power_2 / \
                (2 * grads_power_2 + \
                sum_activations[:, :, None, None] * grads_power_3 + eps)
        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = np.where(gradients != 0, aij, 0)

        weights = np.maximum(gradients, 0) * aij
        weights = np.sum(weights, axis=(2, 3))

        return weights


class XGradCAM(BaseCAM):
    def __init__(self, imgs: np.ndarray, CLASSES: List[str]) -> None:
        super(XGradCAM, self).__init__(imgs, CLASSES)

    def get_cam_weights(self, 
            activations:np.ndarray, 
            gradients:np.ndarray) -> np.ndarray:
        """
            TODO
        """
        sum_activations = np.sum(activations, axis=(2, 3))
        eps = 1e-7
        weights = gradients * activations / \
            (sum_activations[:, :, None, None] + eps)
        weights = weights.sum(axis=(2, 3))
        return weights


class EigenGradCAM(BaseCAM):
    def __init__(self, imgs: np.ndarray, CLASSES: List[str]) -> None:
        super(EigenGradCAM, self).__init__(imgs, CLASSES)

    def get_cam_weights(self, 
        activations:np.ndarray, 
        gradients:np.ndarray) -> np.ndarray:
        return None

    def get_cam_image(self, 
            activations:np.ndarray, 
            gradients:np.ndarray,
            weights:np.ndarray) -> np.ndarray:
        """
            TODO
        """
        return self.get_2d_projection(gradients * activations)
        
    def get_2d_projection(self, activation_batch):
        # TBD: use pytorch batch svd implementation
        activation_batch[np.isnan(activation_batch)] = 0
        projections = []
        for activations in activation_batch:
            reshaped_activations = (activations).reshape(
                activations.shape[0], -1).transpose()
            # Centering before the SVD seems to be important here,
            # Otherwise the image returned is negative
            reshaped_activations = reshaped_activations - \
                reshaped_activations.mean(axis=0)
            U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
            projection = reshaped_activations @ VT[0, :]
            projection = projection.reshape(activations.shape[1:])
            projections.append(projection)
        return np.float32(projections)


class LayerCAM(EigenGradCAM):
    def __init__(self, imgs: np.ndarray, CLASSES: List[str]) -> None:
        super(LayerCAM, self).__init__(imgs, CLASSES)

    def get_cam_image(self, 
            activations:np.ndarray, 
            gradients:np.ndarray,
            weights:np.ndarray,
            eigen_smooth:bool = True) -> np.ndarray:
        """
            TODO
        """
        spatial_weighted_activations = np.maximum(gradients, 0) * activations

        if eigen_smooth:
            cam = self.get_2d_projection(spatial_weighted_activations)
        else:
            cam = spatial_weighted_activations.sum(axis=1)
        return cam
        

def t2n(t):
    return t.detach().cpu().numpy().astype(np.float)


def plt2cvMat(fig):
    ''' matplotlib.figure.Figure转为np.ndarray '''
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf_ndarray = np.frombuffer(fig.canvas.tostring_rgb(), dtype="u1")
    im = buf_ndarray.reshape(h, w, 3)
    im = im[..., ::-1]
    return im