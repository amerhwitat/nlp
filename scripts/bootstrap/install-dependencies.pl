#!/usr/bin/env perl
use strict; use warnings; use File::Basename qw(dirname); use Cwd qw(abs_path); use File::Spec;
my $root=abs_path(File::Spec->catdir(dirname(__FILE__),'..','..')); chdir $root or die "Cannot enter $root: $!";
sub run { system(@_)==0 or die "Command failed: @_\n"; }
my $npm=$^O eq 'MSWin32' ? 'npm.cmd' : 'npm';
run('git','--version'); run($^X,'--version'); run($npm,'--version');
if(-f 'web/package-lock.json'){ run($npm,'ci'); } elsif(-f 'web/package.json'){ run($npm,'install'); }
if(-f 'python/requirements.txt'){
  my $py=$^X; run($py,'-m','venv','.venv') unless -d '.venv';
  my $venv=$^O eq 'MSWin32' ? '.venv/Scripts/python.exe' : '.venv/bin/python';
  run($venv,'-m','pip','install','-r','python/requirements.txt');
}
print "Bootstrap completed.\n";
