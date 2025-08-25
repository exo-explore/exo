package lib

import (
	"log"
	"os"
	"syscall"

	"golang.org/x/sys/unix"
)

func Assert(b bool, msg string) {
	if !b {
		log.Panic(msg)
	}
}

func FstatGetMode(fd int) (os.FileMode, error) {
	// perform fstat syscall
	var sys unix.Stat_t = unix.Stat_t{}
	if err := unix.Fstat(fd, &sys); err != nil {
		return 0, err
	}

	// reconstruct FileMode from sys-struct; SEE: https://github.com/golang/go/blob/5a56d8848b4ffb79c5ccc11ec6fa01823a91aaf8/src/os/stat_linux.go#L17
	mode := os.FileMode(sys.Mode & 0777)
	switch sys.Mode & syscall.S_IFMT {
	case syscall.S_IFBLK:
		mode |= os.ModeDevice
	case syscall.S_IFCHR:
		mode |= os.ModeDevice | os.ModeCharDevice
	case syscall.S_IFDIR:
		mode |= os.ModeDir
	case syscall.S_IFIFO:
		mode |= os.ModeNamedPipe
	case syscall.S_IFLNK:
		mode |= os.ModeSymlink
	case syscall.S_IFREG:
		// nothing to do
	case syscall.S_IFSOCK:
		mode |= os.ModeSocket
	}
	if sys.Mode&syscall.S_ISGID != 0 {
		mode |= os.ModeSetgid
	}
	if sys.Mode&syscall.S_ISUID != 0 {
		mode |= os.ModeSetuid
	}
	if sys.Mode&syscall.S_ISVTX != 0 {
		mode |= os.ModeSticky
	}
	return mode, nil
}
