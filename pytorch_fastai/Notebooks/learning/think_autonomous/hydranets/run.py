import models
import torch
import os

# ScriptModule Feature to carry the model to cpp
# img_var = Variable(torch.from_numpy(prepare_img(img).transpose(2,0,1)[None]), requires_grad = False).float()
# if torch.cuda.is_available():
#     img_var = img_var.cuda()

# traced_script_module = torch.jit.trace(hydranet, img_var)

# traced_script_module.save("hydranet.pt")

if __name__ == '__main__':

    isTrace = True

    hydranet =models.net(num_classes=6, num_tasks=2)

    hydranet.eval()

    if isTrace:
        __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

        weights_location = os.path.join(__location__,'ExpKITTI_joint.ckpt')
        
        ckpt = torch.load(weights_location)
        hydranet.load_state_dict(ckpt['state_dict'])

        traced_script_module = torch.jit.trace(hydranet, torch.rand(1, 3, 1224, 370))

        print(traced_script_module.code)

        traced_script_module.save("traced_hydranet.pt")

    else:

        __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

        weights_location = os.path.join(__location__,'ExpKITTI_joint.ckpt')
        
        ckpt = torch.load(weights_location)
        hydranet.load_state_dict(ckpt['state_dict'])

        script_module = torch.jit.script(hydranet)

        script_module.save("annotated_hydranet.pt")

