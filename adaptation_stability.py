"""Opt-in BN statistics control and sampled, non-mutating gradient diagnostics."""

def freeze_backbone_bn_statistics(feature):
    import torch
    count=0
    for module in feature.modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            module.eval();count+=1
    # eval freezes running statistics only; affine parameters retain gradients.
    return count


def shared_gradient_metrics(source_loss,ssl_loss,parameters):
    import torch
    params=tuple(p for p in parameters if p.requires_grad)
    source=torch.autograd.grad(source_loss,params,retain_graph=True,allow_unused=True)
    ssl=torch.autograd.grad(ssl_loss,params,retain_graph=True,allow_unused=True)
    device=source_loss.device
    a=torch.zeros((),device=device);b=a.clone();dot=a.clone()
    for gs,gt in zip(source,ssl):
        if gs is not None:a+=gs.detach().float().square().sum()
        if gt is not None:b+=gt.detach().float().square().sum()
        if gs is not None and gt is not None:dot+=(gs.detach().float()*gt.detach().float()).sum()
    sn=a.sqrt().item();tn=b.sqrt().item()
    return dict(source_grad_norm=sn,weighted_ssl_grad_norm=tn,
        ssl_to_source_grad_ratio=tn/sn if sn>0 else None,
        grad_cosine=dot.item()/(sn*tn) if sn>0 and tn>0 else None)
