class NMSMultiTarget(nn.Module):
	def ___init__(self, target_thresh):
		super(NMSMultiTarget, self).__init__()
		self._target_thresh = target_thresh
		self._num_thresh = len(target_thresh)
	
	def forward(self, training, )
