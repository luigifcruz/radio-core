def lfilter(self, b, a, x, zi):
    if self.cuda:
        x = self.xp.asnumpy(x)

    x, zi = self.ss.lfilter(b, a, x, zi=zi)

    if self.cuda:
        x = self.xp.asarray(x)

    return (x, zi)


def filtfilt(self, b, a, x):
    if self.cuda:
        x = self.xp.asnumpy(x)

    x = self.ss.filtfilt(b, a, x)

    if self.cuda:
        x = self.xp.asarray(x)

    return x
