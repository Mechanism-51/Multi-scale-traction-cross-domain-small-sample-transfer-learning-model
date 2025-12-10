import torch

def CORAL_loss(source, target):
	d = source.data.shape[1]
	ns, nt = source.data.shape[0], target.data.shape[0]
	xm = torch.mean(source, 0, keepdim=True) - source
	xc = xm.t() @ xm / (ns - 1)

	xmt = torch.mean(target, 0, keepdim=True) - target
	xct = xmt.t() @ xmt / (nt - 1)

	loss = torch.mul((xc - xct), (xc - xct))
	loss = torch.sum(loss) / (4 * d * d)
	return loss
