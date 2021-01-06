dis_fake_loss_list = []
dis_real_loss_list = []
dis_loss_list = []
gen_loss_list = []

for epoch in range(hparams['epochs']):
    for i, data in enumerate(real_dataloader, 0):

        dis_model.zero_grad()

        dis_model = dis_model.to(device)
        real_label = real_label.to(device)

        data = data[0]
        data = data.to(device)

        real_outputs = dis_model(data)
        real_loss = loss_function(real_outputs, real_label)
        real_loss.backward()
        running_real_loss = real_loss.item()
        dis_real_loss.append(running_real_loss)

        data = data.to("cpu")

        del data
        #dis_model = dis_model.to("cpu")

        fake_noise = torch.randn(
            hparams['batch_size'], hparams['z_shape'], 1, 1)

        gen_model = gen_model.to(device)
        fake_label = fake_label.to(device)
        fake_noise = fake_noise.to(device)

        fake_image = gen_model(fake_noise)

        #fake_noise = fake_noise.to("cpu")
        #gen_model = gen_model.to("cpu")

        dis_model = dis_model.to(device)
        fake_outputs = dis_model(fake_image.detach())

        fake_loss = loss_function(fake_outputs, fake_label)
        fake_loss.backward()
        running_fake_loss = fake_loss.item()
        dis_fake_loss.append(running_fake_loss)
        dis_loss_list.append(running_fake_loss+running_real_loss)

        dis_optimizer.step()

        gen_model.zero_grad()
        fake_outputs = dis_model(fake_image.detach())

        gen_loss = loss_function(fake_outputs, real_label)
        gen_loss.backward()
        running_gen_loss = fake_loss.item()
        gen_loss_list.append(running_gen_loss)
        gen_optimizer.step()
