# -*- mode: ruby -*-
# vi: set ft=ruby :
# About: Vagrant file for the Digital Twin environment

###############
#   Modules   #
###############

module OS
  def OS.windows?
      (/cygwin|mswin|mingw|bccwin|wince|emx/ =~ RUBY_PLATFORM) != nil
  end

  def OS.mac?
      (/darwin/ =~ RUBY_PLATFORM) != nil
  end

  def OS.unix?
      !OS.windows?
  end

  def OS.linux?
      OS.unix? and not OS.mac?
  end
end


###############
#  Variables  #
###############

CPUS = 2
# Increased RAM for PyTorch and ML dependencies
RAM = 6144

# Bento: Packer templates for building minimal Vagrant baseboxes
BOX = "bento/ubuntu-20.04"
VM_NAME = "ubuntu-20.04-comnetsemu-dt"

# When using libvirt as the provider, use this box
BOX_LIBVIRT = "generic/ubuntu2204"

######################
#  Provision Script  #
######################

# OS identification
if OS.windows?
  puts "Vagrant launched from windows."
elsif OS.unix?
  puts "Vagrant launched from unix."
else
  puts "Vagrant launched from unknown platform."
end

# Common bootstrap
$bootstrap= <<-SCRIPT
DEBIAN_FRONTEND=noninteractive echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf > /dev/null
DEBIAN_FRONTEND=noninteractive systemctl restart systemd-resolved
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get upgrade -y

APT_PKGS=(
  ansible
  bash-completion
  dfc
  gdb
  git
  htop
  iperf
  iperf3
  make
  pkg-config
  python3
  python3-dev
  python3-pip
  sudo
  tmux
)
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${APT_PKGS[@]}"
SCRIPT

if OS.windows?
puts "Changing EOL of bash scripts"
$setup_for_windows= <<-SCRIPT
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends dos2unix
DEBIAN_FRONTEND=noninteractive find /home/vagrant/comnetsemu -name "*.sh" -exec dos2unix '{}' '\;'
DEBIAN_FRONTEND=noninteractive find /home/vagrant/Networking_DT4SDN -name "*.sh" -exec dos2unix '{}' '\;'
SCRIPT
end

$setup_x11_server= <<-SCRIPT
APT_PKGS=(
  openbox
  xauth
  xorg
  xterm
)
DEBIAN_FRONTEND=noninteractive apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y "${APT_PKGS[@]}"
SCRIPT

$setup_comnetsemu= <<-SCRIPT
# Apply a custom Xterm profile that looks better than the default.
cp /home/vagrant/comnetsemu/util/Xresources /home/vagrant/.Xresources
# xrdb can not run directly during vagrant up. Auto-works after reboot.
xrdb -merge /home/vagrant/.Xresources

cd /home/vagrant/comnetsemu/util || exit
bash ./install.sh -a

# Run the custom shell script, if it exists.
cd /home/vagrant/comnetsemu/util || exit
if [ -f "./vm_customize.sh" ]; then
  echo "*** Run VM customization script."
  bash ./vm_customize.sh
fi
SCRIPT

# NEW: Setup for Digital Twin specific dependencies
$setup_digital_twin= <<-SCRIPT
echo "*** Setting up Digital Twin environment ***"

# Clone the Digital Twin repository
cd /home/vagrant || exit
if [ ! -d "Networking_DT4SDN" ]; then
  git clone https://github.com/MattDema/Networking_DT4SDN.git
  chown -R vagrant:vagrant Networking_DT4SDN
fi

# Run the Digital Twin setup script
cd /home/vagrant/Networking_DT4SDN || exit
if [ -f "setups/setup_vm.sh" ]; then
  chmod +x setups/setup_vm.sh
  sudo -u vagrant bash ./setups/setup_vm.sh
  echo "*** Digital Twin setup completed ***"
else
  echo "WARNING: setup_vm.sh not found in setups/ directory"
fi
SCRIPT

$post_installation= <<-SCRIPT
# Allow the vagrant user to use Docker without sudo.
usermod -aG docker vagrant
if [ -d /home/vagrant/.docker ]; then
  chown -R vagrant:vagrant /home/vagrant/.docker
fi
DEBIAN_FRONTEND=noninteractive apt-get autoclean -y
DEBIAN_FRONTEND=noninteractive apt-get autoremove -y

echo "*** Post-installation completed ***"
echo "*** Digital Twin VM is ready ***"
SCRIPT

$setup_libvirt_vm_always= <<-SCRIPT
# Configure the SSH server to allow X11 forwarding with sudo
cp /home/vagrant/comnetsemu/util/sshd_config /etc/ssh/sshd_config
systemctl restart sshd.service
SCRIPT

####################
#  Vagrant Config  #
####################

Vagrant.configure("2") do |config|

  config.ssh.forward_x11 = true
  config.vm.boot_timeout = 900
  config.vm.define "comnetsemu_dt" do |comnetsemu_dt|
    comnetsemu_dt.vm.box = BOX
    # Sync ./ to home directory of vagrant to simplify the install script
    comnetsemu_dt.vm.synced_folder ".", "/vagrant", disabled: true
    comnetsemu_dt.vm.synced_folder ".", "/home/vagrant/comnetsemu"

    # For Virtualbox provider
    comnetsemu_dt.vm.provider "virtualbox" do |vb|
      vb.name = VM_NAME
      vb.cpus = CPUS
      vb.memory = RAM
      # MARK: The vCPUs should have SSE4 to compile DPDK applications.
      vb.customize ["setextradata", :id, "VBoxInternal/CPUM/SSE4.1", "1"]
      vb.customize ["setextradata", :id, "VBoxInternal/CPUM/SSE4.2", "1"]
    end

    # For libvirt provider
    comnetsemu_dt.vm.provider "libvirt" do |libvirt, override|
      override.vm.box = BOX_LIBVIRT
      override.vm.synced_folder ".", "/home/vagrant/comnetsemu", type: "nfs", nfs_version: 4

      libvirt.driver = "kvm"
      libvirt.cpus = CPUS
      libvirt.memory = RAM
    end

    comnetsemu_dt.vm.hostname = "comnetsemu-dt"
    comnetsemu_dt.vm.box_check_update = true
    comnetsemu_dt.vm.post_up_message = '
The Digital Twin VM is up! Run "$ vagrant ssh comnetsemu_dt" to ssh into the running VM.

This VM has been configured with:
- 6GB RAM for PyTorch and ML dependencies
- ComNetsEmu framework installed
- Digital Twin repository cloned at /home/vagrant/Networking_DT4SDN

To connect to the Physical Twin:
1. Set the Physical Twin IP: export PT_IP=192.168.56.101
2. Start the orchestrator: cd ~/Networking_DT4SDN && python3 src/controllers/orchestrator.py
    '

    comnetsemu_dt.vm.provision :shell, inline: $bootstrap, privileged: true
    comnetsemu_dt.vm.provision :shell, inline: $setup_x11_server, privileged: true
    if OS.windows?
      comnetsemu_dt.vm.provision :shell, inline: $setup_for_windows, privileged: true
    end
    comnetsemu_dt.vm.provision :shell, inline: $setup_comnetsemu, privileged: false
    comnetsemu_dt.vm.provision :shell, inline: $setup_digital_twin, privileged: true
    comnetsemu_dt.vm.provision :shell, inline: $post_installation, privileged: true

    comnetsemu_dt.vm.provider "libvirt" do |libvirt, override|
      override.vm.provision :shell, inline: $setup_libvirt_vm_always, privileged: true, run: "always"
    end
    comnetsemu_dt.vm.provision :shell, privileged:false, run: "always", path: "./util/echo_banner.sh"

    # VM networking - DIFFERENT PORTS to avoid conflicts with Physical Twin
    comnetsemu_dt.vm.network "forwarded_port", guest: 8888, host: 8889, host_ip: "127.0.0.1"
    comnetsemu_dt.vm.network "forwarded_port", guest: 8082, host: 8092
    comnetsemu_dt.vm.network "forwarded_port", guest: 8083, host: 8093
    comnetsemu_dt.vm.network "forwarded_port", guest: 8084, host: 8094

    # Private network to communicate with Physical Twin
    comnetsemu_dt.vm.network "private_network", ip: "192.168.56.102"

    # Enable X11 forwarding
    comnetsemu_dt.ssh.forward_agent = true
    comnetsemu_dt.ssh.forward_x11 = true
  end
end
