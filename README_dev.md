# Play-With-Pytorch
## <span style="color:red">[Dev Branch]</span>
![alt text](https://pytorch.org/tutorials/_images/pytorch-logo-flat.png "Pytorch Logo")

**Play With Pytorch - By Sameera Sandaruwan**

## Problems;

* **Transformer Resize problem - Resize function change the smallest size from H or W to the given scale.**

## Notes;

* **Loaders are slow. That's why training is slow.** > Use `pin_memory=True`

```
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

## Visualise NN using;
[Netron](https://github.com/lutzroeder/netron)

Run `pip install netron`
Then `netron [FILE]` or `import netron`; `netron.start('[FILE]')`

## To Do;
- [ ] **Trainer** - use of different **criterions - Why `reduction` in validation?**
- [x] **Nvidia Drivers NOT available** - https://www.nvidia.com/download/driverResults.aspx/135394/en-us
- [ ] **Trainer class** - Finish TODOs - https://keras.io/models/sequential/
- [ ] Custom dataset - [SubsetRandomSampler](https://pytorch.org/docs/master/data.html#torch.utils.data.SubsetRandomSampler)
- [ ] **Play with RL** 
- [ ] without split - two folders for training and test data is already supplied
- [ ] 

## Done;
- [x] use opencv to read images and see the speed
- [x] bass_util - Image Resizer and Save function
- [x] File crawler
- [x] Files to tensor converter
- [x] Resize images
- [x] Custom dataset - split dataset into, train, valid, test = [random split](https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split) or [SubsetRandomSampler](https://pytorch.org/docs/master/data.html#torch.utils.data.SubsetRandomSampler)
- [x] data loader
- [x] How to normalize
- [x] Create model
- [x] bass_util > cmd logging function
- [x] save loss data to JSON
- [x] A way to print 10% of the batches while training
- [x] Handling high res. image - inspire VGG16 Pytorch
